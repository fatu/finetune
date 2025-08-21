import torch
torch.set_float32_matmul_precision('high')
import json
import re
from typing import List, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from math_verify import LatexExtractionConfig, parse, verify


@dataclass
class EvalConfig:
    """Configuration for model evaluation."""
    model_name: str = "Qwen/Qwen3-4B"
    dataset_name: str = "opencompass/AIME2025"  # OpenCompass AIME 2025 dataset
    dataset_config: str = "AIME2025-I"  # Config name: AIME2025-I or AIME2025-II
    max_length: int = 2048
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    batch_size: int = 1
    tensor_parallel_size: int = 1  # Number of GPUs for tensor parallelism
    gpu_memory_utilization: float = 0.9  # GPU memory utilization
    output_file: str = "aime_2025_eval_results.json"
    num_samples: int = -1  # -1 for all samples
    thinking_budget: int = -1  # Max tokens in <think> section, -1 for unlimited
    stop_at_think_close: bool = True  # Stop generation at </think> tag


class AIMEEvaluatorVLLM:
    """Evaluator for AIME math problems using vLLM."""
    
    SYSTEM_PROMPT = (
        "You are a mathematical problem solver. Solve the given problem step by step. "
        "First, think through your reasoning process, then provide the final answer. "
        "Format your response as:\n"
        "<think>Your step-by-step reasoning here</think>\n"
        "Final Answer: Your final numerical answer here"
    )
    
    def __init__(self, config: EvalConfig):
        self.config = config
        
        # Load tokenizer for prompt formatting
        print(f"Loading tokenizer: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize vLLM model
        print(f"Loading vLLM model: {config.model_name}")
        self.llm = LLM(
            model=config.model_name,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_length + config.max_new_tokens,
            trust_remote_code=True,  # Required for some models
        )
        
        # Set up sampling parameters
        stop_tokens = []
        if config.stop_at_think_close:
            stop_tokens.append("</think>")
        
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens,
            stop=stop_tokens if stop_tokens else None,
        )
        
        # Sampling params for thinking phase (if budget is set)
        if config.thinking_budget > 0:
            self.thinking_sampling_params = SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.thinking_budget,
                stop=["</think>"],
            )
        else:
            self.thinking_sampling_params = None
        
    def format_prompt(self, problem: str) -> str:
        """Format the problem into a prompt for the model."""
        # Check if model is Gemma (Gemma models don't support system role)
        if "gemma" in self.config.model_name.lower():
            # Gemma format: combine system prompt with user message
            combined_content = f"{self.SYSTEM_PROMPT}\n\n{problem}"
            messages = [
                {"role": "user", "content": combined_content}
            ]
        else:
            # Standard format for Qwen and other models
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": problem}
            ]
        
        # Try to apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except:
                # Fallback for models without proper chat template
                if "gemma" in self.config.model_name.lower():
                    prompt = f"<bos>{combined_content}\n\nAssistant:"
                else:
                    prompt = f"{self.SYSTEM_PROMPT}\n\nProblem: {problem}\n\nSolution:"
        else:
            # Fallback to simple formatting
            prompt = f"{self.SYSTEM_PROMPT}\n\nProblem: {problem}\n\nSolution:"
        return prompt
    
    def generate_solutions_batch(self, problems: List[str]) -> List[str]:
        """Generate solutions for a batch of problems using vLLM."""
        # Format all prompts
        prompts = [self.format_prompt(problem) for problem in problems]
        
        # If thinking budget is set, we need to do two-phase generation
        if self.thinking_sampling_params is not None:
            return self._generate_with_thinking_budget(prompts)
        
        # Standard generation without thinking budget
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        # Extract generated text
        generated_texts = []
        for output in outputs:
            generated_texts.append(output.outputs[0].text)
        
        return generated_texts
    
    def _generate_with_thinking_budget(self, prompts: List[str]) -> List[str]:
        """Generate with controlled thinking budget using two-phase generation."""
        final_outputs = []
        
        for prompt in prompts:
            # Phase 1: Generate thinking with budget
            thinking_prompt = prompt + "<think>"
            thinking_output = self.llm.generate([thinking_prompt], self.thinking_sampling_params)[0]
            thinking_text = thinking_output.outputs[0].text
            
            # Check if thinking naturally ended with </think>
            if not thinking_text.endswith("</think>"):
                thinking_text += "</think>"
            
            # Phase 2: Generate final answer
            full_prompt = thinking_prompt + thinking_text + "\nFinal Answer: "
            final_output = self.llm.generate(
                [full_prompt], 
                SamplingParams(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=256,  # Final answer should be short
                )
            )[0]
            
            # Combine thinking and final answer
            complete_response = "<think>" + thinking_text + "\nFinal Answer: " + final_output.outputs[0].text
            final_outputs.append(complete_response)
        
        return final_outputs
    
    def extract_answer(self, solution: str) -> str:
        """Extract the numerical answer from the solution."""
        # Try to extract answer from <answer> tags
        answer_match = re.search(r'<answer>(.*?)</answer>', solution, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        
        # Look for "Final Answer" and get the first number after it
        final_answer_match = re.search(r'final\s+answer[:\s]*.*?(\d+(?:\.\d+)?)', solution, re.IGNORECASE | re.DOTALL)
        if final_answer_match:
            return final_answer_match.group(1)
        
        # Fallback: look for other answer patterns
        patterns = [
            r'(?:answer|solution)(?:\s+is)?[:\s]+(\d+(?:\.\d+)?)',
            r'(?:therefore|thus|hence)[,\s]+.*?(\d+(?:\.\d+)?)',
            r'=\s*(\d+(?:\.\d+)?)\s*$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)
        
        # Last resort: find the last number in the solution
        numbers = re.findall(r'\d+(?:\.\d+)?', solution)
        if numbers:
            return numbers[-1]
        
        return "no answer found"
    
    def verify_answer(self, predicted: str, ground_truth: str) -> bool:
        """Verify if the predicted answer matches the ground truth."""
        try:
            # Parse both answers using math_verify
            pred_parsed = parse(predicted, extraction_mode="first_match", 
                              extraction_config=[LatexExtractionConfig()])
            truth_parsed = parse(ground_truth, extraction_mode="first_match",
                               extraction_config=[LatexExtractionConfig()])
            
            if len(truth_parsed) > 0:
                return bool(verify(pred_parsed, truth_parsed))
            else:
                # Fallback to simple string/numeric comparison
                pred_val = float(predicted.strip())
                truth_val = float(ground_truth.strip())
                return abs(pred_val - truth_val) < 1e-6
        except:
            # Direct string comparison as last resort
            return predicted.strip() == ground_truth.strip()
    
    def evaluate_dataset(self, dataset) -> Dict[str, Any]:
        """Evaluate the model on the entire dataset using vLLM batch processing."""
        results = {
            "model": self.config.model_name,
            "dataset": f"{self.config.dataset_name}/{self.config.dataset_config}",
            "total_problems": 0,
            "correct": 0,
            "accuracy": 0.0,
            "problems": []
        }
        
        # Limit samples if specified
        if self.config.num_samples > 0:
            dataset = dataset.select(range(min(self.config.num_samples, len(dataset))))
        
        # Process in batches
        batch_size = self.config.batch_size
        num_batches = (len(dataset) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            
            # Prepare batch
            batch_problems = []
            batch_ground_truths = []
            batch_indices = []
            
            for idx in range(start_idx, end_idx):
                example = dataset[idx]
                problem = example.get("problem", example.get("question", ""))
                ground_truth = example.get("solution", example.get("answer", ""))
                
                batch_problems.append(problem)
                batch_ground_truths.append(ground_truth)
                batch_indices.append(idx)
            
            # Generate solutions for the batch
            generated_solutions = self.generate_solutions_batch(batch_problems)
            
            # Process results
            for i, (problem, generated_solution, ground_truth, idx) in enumerate(
                zip(batch_problems, generated_solutions, batch_ground_truths, batch_indices)
            ):
                predicted_answer = self.extract_answer(generated_solution)
                
                # Extract ground truth answer if it's in solution format
                if "<answer>" in ground_truth:
                    ground_truth_answer = self.extract_answer(ground_truth)
                else:
                    ground_truth_answer = ground_truth
                
                # Verify answer
                is_correct = self.verify_answer(predicted_answer, ground_truth_answer)
                
                # Store result
                problem_result = {
                    "id": idx,
                    "problem": problem[:200] + "..." if len(problem) > 200 else problem,
                    "generated_solution": generated_solution,
                    "predicted_answer": predicted_answer,
                    "ground_truth_answer": ground_truth_answer,
                    "correct": is_correct
                }
                
                results["problems"].append(problem_result)
                results["total_problems"] += 1
                if is_correct:
                    results["correct"] += 1
            
            # Print progress
            if (batch_idx + 1) % 5 == 0:
                current_acc = results["correct"] / results["total_problems"]
                print(f"Progress: {results['total_problems']}/{len(dataset)}, Current Accuracy: {current_acc:.2%}")
        
        # Calculate final accuracy
        results["accuracy"] = results["correct"] / results["total_problems"] if results["total_problems"] > 0 else 0.0
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate language models on AIME 2025 dataset using vLLM")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B", 
                       help="Model name or path (e.g., Qwen/Qwen3-4B, google/gemma-3-270m)")
    parser.add_argument("--dataset", type=str, default="opencompass/AIME2025",
                       help="Dataset name or path")
    parser.add_argument("--dataset_config", type=str, default="AIME2025-I",
                       help="Dataset config (AIME2025-I or AIME2025-II)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=-1,
                       help="Number of samples to evaluate (-1 for all)")
    parser.add_argument("--output", type=str, default="aime_2025_eval_results.json",
                       help="Output file for results")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p for generation")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                       help="Maximum new tokens to generate")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory utilization (0-1)")
    parser.add_argument("--thinking_budget", type=int, default=-1,
                       help="Maximum tokens for thinking section (-1 for unlimited)")
    parser.add_argument("--no_stop_at_think", action="store_true",
                       help="Don't stop at </think> tag (continue generating)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = EvalConfig(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        output_file=args.output,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        thinking_budget=args.thinking_budget,
        stop_at_think_close=not args.no_stop_at_think
    )
    
    # Initialize evaluator
    evaluator = AIMEEvaluatorVLLM(config)
    
    # Load dataset
    print(f"Loading dataset: {config.dataset_name} with config: {config.dataset_config}")
    try:
        # Try to load AIME 2025 dataset from OpenCompass with config
        dataset = load_dataset(config.dataset_name, config.dataset_config, split="test")
    except Exception as e:
        # Fallback to a general math dataset for testing
        print(f"Could not load {config.dataset_name}: {e}")
        print("Using AI-MO/NuminaMath-TIR for testing instead")
        dataset = load_dataset("AI-MO/NuminaMath-TIR", split="test[:100]")
    
    # Run evaluation
    print("Starting evaluation...")
    results = evaluator.evaluate_dataset(dataset)
    
    # Save results
    with open(config.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {results['model']}")
    print(f"Dataset: {results['dataset']}")
    print(f"Total Problems: {results['total_problems']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Results saved to: {config.output_file}")


if __name__ == "__main__":
    main()