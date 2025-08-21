import torch
torch.set_float32_matmul_precision('high')
import json
import re
from typing import List, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    batch_size: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_file: str = "aime_2025_eval_results.json"
    num_samples: int = -1  # -1 for all samples


class AIMEEvaluator:
    """Evaluator for AIME math problems."""
    
    SYSTEM_PROMPT = (
        "You are a mathematical problem solver. Solve the given problem step by step. "
        "First, think through your reasoning process, then provide the final answer. "
        "Format your response as:\n"
        "<think>Your step-by-step reasoning here</think>\n"
        "<answer>Your final numerical answer here</answer>"
    )
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load model and tokenizer
        print(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype="auto",
            device_map="auto" if config.device == "cuda" else None
        )
        self.model.eval()
        
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
    
    def generate_solution(self, problem: str) -> str:
        """Generate a solution for the given problem."""
        prompt = self.format_prompt(problem)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated
    
    def extract_answer(self, solution: str) -> str:
        """Extract the numerical answer from the solution."""
        # Try to extract answer from <answer> tags
        answer_match = re.search(r'<answer>(.*?)</answer>', solution, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        
        # Fallback: look for final numerical answer patterns
        patterns = [
            r'(?:answer|solution)(?:\s+is)?[:\s]+(\d+(?:\.\d+)?)',
            r'(?:therefore|thus|hence)[,\s]+.*?(\d+(?:\.\d+)?)',
            r'=\s*(\d+(?:\.\d+)?)\s*$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)
        
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
        """Evaluate the model on the entire dataset."""
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
        
        for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
            problem = example.get("problem", example.get("question", ""))
            ground_truth = example.get("solution", example.get("answer", ""))
            
            # Generate solution
            generated_solution = self.generate_solution(problem)
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
            if (idx + 1) % 10 == 0:
                current_acc = results["correct"] / results["total_problems"]
                print(f"Progress: {idx+1}/{len(dataset)}, Current Accuracy: {current_acc:.2%}")
            # TODO remove break
            break
        
        # Calculate final accuracy
        results["accuracy"] = results["correct"] / results["total_problems"] if results["total_problems"] > 0 else 0.0
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate language models on AIME 2025 dataset")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B", 
                       help="Model name or path (e.g., Qwen/Qwen3-4B, google/gemma-3-270m)")
    parser.add_argument("--dataset", type=str, default="opencompass/AIME2025",
                       help="Dataset name or path")
    parser.add_argument("--dataset_config", type=str, default="AIME2025-I",
                       help="Dataset config (AIME2025-I or AIME2025-II)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=-1,
                       help="Number of samples to evaluate (-1 for all)")
    parser.add_argument("--output", type=str, default="aime_2025_eval_results.json",
                       help="Output file for results")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p for generation")
    
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
        top_p=args.top_p
    )
    
    # Initialize evaluator
    evaluator = AIMEEvaluator(config)
    
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