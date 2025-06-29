# grpo_appworld.py
"""
GRPO training with AppWorld environment for interactive task learning.
This script uses the AppWorld environment to generate prompts and evaluate completions.

Key Features:
- Loads tasks directly from AppWorld environment instead of offline datasets
- Executes generated code in the AppWorld environment for reward computation
- Supports multiple reward functions (task completion, syntax validity, API usage)
- Configurable task count and environment interaction limits

Usage:
    # Basic training with LoRA
    python finetune/grpo_appworld.py
    
    # Custom configuration
    python finetune/grpo_appworld.py --model_path model/Qwen/Qwen3-0.6B \
                                     --max_tasks 50 \
                                     --max_interactions 20
    
    # Training without LoRA (full fine-tuning)
    # Note: You would need to modify the script to accept command-line args
    # or directly edit the use_lora parameter in the main function

Requirements:
    - AppWorld server must be running (default: http://0.0.0.0:8081)
    - Install AppWorld: pip install appworld
    - Start server: appworld serve
"""

from datasets import Dataset
from trl_main import GRPOEnvConfig, GRPOEnvTrainer
from finetune.environment.appworld_env import AppWorldEnv, build_appworld_envs
from finetune.prompts.appworld import APPWORLD_SYSTEM_PROMPT_NO_HIS_NO_THINK, USER_PROMPT
from typing import List, Dict, Any, Optional
import torch
import json
import logging
from peft import LoraConfig


class AppWorldDataset:
    """Custom dataset that yields prompts from AppWorld environment."""
    
    def __init__(self, dataset_name="train", max_tasks=None):
        self.dataset_name = dataset_name
        self.max_tasks = max_tasks
        self.data = []
        self.environments = {}  # Map task_id to actual environment objects
        self._load_data()
    
    def _load_data(self):
        """Load task prompts from AppWorld environment."""
        envs = build_appworld_envs(dataset_name=self.dataset_name)
        count = 0
        for task_id, env in envs:
            if self.max_tasks and count >= self.max_tasks:
                break
            
            # Reset environment and get initial observation
            obs, info = env.reset(task_id)
            
            # Extract supervisor info
            supervisor = info.get("supervisor", {})
            
            # Format the user prompt with supervisor details and task
            user_content = USER_PROMPT.format(
                supervisor_first_name=supervisor.get("first_name", "John"),
                supervisor_last_name=supervisor.get("last_name", "Doe"),
                supervisor_email=supervisor.get("email", "john.doe@example.com"),
                supervisor_phone_number=supervisor.get("phone_number", "+1234567890"),
                task_description=obs
            )
            
            # Create a prompt in conversation format
            prompt = [
                {
                    "role": "system",
                    "content": APPWORLD_SYSTEM_PROMPT_NO_HIS_NO_THINK
                },
                {
                    "role": "user",
                    "content": user_content + " /no_think"
                }
            ]
            
            # Store the data with task metadata
            self.data.append({
                "prompt": prompt,
                "task_id": task_id,
                "task_instruction": obs,
                "supervisor": supervisor
            })
            
            # Store the actual environment object (not in dataset, but in this class)
            self.environments[task_id] = env
            # Don't close the environment - we'll use it later
            
            count += 1
    
    def to_dataset(self):
        """Convert to HuggingFace Dataset format."""
        return Dataset.from_list(self.data)
    
    def get_environment(self, task_id):
        """Get the environment for a specific task."""
        return self.environments.get(task_id)
    
    def __del__(self):
        """Clean up environments when the dataset is destroyed."""
        for env in self.environments.values():
            try:
                env.close()
            except:
                pass


# AppWorld-specific reward functions

class AppWorldRewardFunction:
    """Reward function that executes generated code in AppWorld environment."""
    
    def __init__(self, max_interactions=30, dataset_name="train"):
        self.max_interactions = max_interactions
        self.dataset_name = dataset_name
        self.logger = logging.getLogger(__name__)
        self.__name__ = "AppWorldRewardFunction"
    
    def __call__(self, completions: List[List[Dict]], prompts: List[Any], 
                 task_ids: List[str] = None, **kwargs) -> List[float]:
        """
        Execute generated code in AppWorld environment and compute rewards.
        
        Args:
            completions: List of generated completions for each prompt
            prompts: Original prompts (not used here but required by interface)
            task_ids: Task IDs from the dataset
            **kwargs: Additional data from the dataset (including 'env' if available)
            
        Returns:
            List of rewards for each completion
        """
        rewards = []
        
        # Extract task_ids and environments from kwargs
        if task_ids is None:
            task_ids = kwargs.get("task_id", [None] * len(completions))
        
        # Get environments from kwargs if available
        environments = kwargs.get("env", [None] * len(completions))
        
        for i, completion_list in enumerate(completions):
            if not completion_list or len(completion_list) == 0:
                rewards.append(0.0)
                continue
                
            # Extract the generated code from completion
            generated_text = completion_list[0]["content"]
            
            # Try to extract Python code from the generated text
            code = self._extract_code(generated_text)
            
            if not code:
                rewards.append(0.0)
                continue
            
            # Get task ID and environment for this completion
            task_id = task_ids[i] if i < len(task_ids) else None
            env = environments[i] if i < len(environments) else None
            
            if task_id is None:
                self.logger.warning(f"No task_id for completion {i}")
                rewards.append(0.0)
                continue
            
            # Execute in AppWorld environment (pass env if available)
            reward = self._execute_in_environment(code, task_id, env)
            rewards.append(reward)
        
        return rewards
    
    def _extract_code(self, text: str) -> Optional[str]:
        """Extract Python code from generated text."""
        # First check for <code> tags (AppWorld format)
        if "<code>" in text and "</code>" in text:
            start = text.find("<code>") + 6
            end = text.find("</code>")
            if end > start:
                return text[start:end].strip()
        
        # Try to find markdown code blocks
        if "```python" in text:
            start = text.find("```python") + 9
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()
        
        # If no code blocks, assume the entire text is code
        # (but filter out obvious non-code)
        if any(keyword in text.lower() for keyword in ["here's", "i'll", "let me", "this code"]):
            # Likely contains explanation, try to extract after newline
            lines = text.split('\n')
            code_lines = []
            in_code = False
            for line in lines:
                if any(keyword in line for keyword in ["def ", "import ", "from ", "class ", "if __name__"]):
                    in_code = True
                if in_code:
                    code_lines.append(line)
            if code_lines:
                return '\n'.join(code_lines)
        
        # Return as-is if it looks like code
        if any(keyword in text for keyword in ["apis.", "import ", "def ", "="]):
            return text
        
        return None
    
    def _execute_in_environment(self, code: str, task_id: str, env=None) -> float:
        """Execute code in AppWorld environment and return reward."""
        try:
            # Use provided environment or create a new one
            if env is None:
                env = AppWorldEnv(max_interactions=self.max_interactions)
                obs, info = env.reset(task_id)
                should_close = True
            else:
                # Environment already exists, just reset it to ensure it's ready
                obs, info = env.reset(task_id)
                should_close = False  # Don't close the shared environment
            
            # Execute the generated code
            obs, reward, done, step_info = env.step(code)
            
            # Only close if we created the environment
            if should_close:
                env.close()
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error executing code for task {task_id}: {e}")
            return 0.0


def syntax_validity_reward(completions: List[List[Dict]], **kwargs) -> List[float]:
    """Reward syntactically valid Python code."""
    import ast
    rewards = []
    
    for completion_list in completions:
        if not completion_list or len(completion_list) == 0:
            rewards.append(0.0)
            continue
            
        text = completion_list[0]["content"]
        
        # Try to extract code
        code = None
        # First check for <code> tags (AppWorld format)
        if "<code>" in text and "</code>" in text:
            start = text.find("<code>") + 6
            end = text.find("</code>")
            if end > start:
                code = text[start:end].strip()
        elif "```python" in text:
            start = text.find("```python") + 9
            end = text.find("```", start)
            if end != -1:
                code = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start) 
            if end != -1:
                code = text[start:end].strip()
        else:
            # Assume entire text is code
            code = text
        
        if not code:
            rewards.append(0.0)
            continue
        
        # Check if it's valid Python
        try:
            ast.parse(code)
            rewards.append(1.0)
        except SyntaxError:
            rewards.append(0.0)
    
    return rewards


def api_usage_reward(completions: List[List[Dict]], **kwargs) -> List[float]:
    """Reward proper usage of AppWorld APIs."""
    rewards = []
    
    for completion_list in completions:
        if not completion_list or len(completion_list) == 0:
            rewards.append(0.0)
            continue
            
        text = completion_list[0]["content"]
        
        # Check for API usage patterns
        api_patterns = ["apis.", "supervisor.complete_task", "execute(", "evaluate("]
        api_count = sum(1 for pattern in api_patterns if pattern in text)
        
        # Normalize to 0-1 range
        reward = min(1.0, api_count * 0.3)
        rewards.append(reward)
    
    return rewards


# Main training script
def train_appworld_model(
    dataset_name="train",
    model_path="model/Qwen/Qwen3-0.6B",
    output_dir="output/appworld-grpo",
    max_tasks=100,
    max_interactions=30,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
):
    """
    Train a model on AppWorld tasks using GRPO with environment feedback.
    
    Args:
        dataset_name: AppWorld dataset split ('train', 'dev', 'test_normal', 'test_challenge')
        model_path: Path to base model
        output_dir: Where to save the trained model
        max_tasks: Maximum number of tasks to train on
        max_interactions: Maximum interactions per task in environment
        use_lora: Whether to use LoRA for parameter-efficient fine-tuning
        lora_r: LoRA rank (dimension of the low-rank matrices)
        lora_alpha: LoRA scaling parameter
        lora_dropout: Dropout probability for LoRA layers
        lora_target_modules: List of module names to apply LoRA to (default: Qwen attention layers)
    """
    
    # Create AppWorld dataset
    print(f"Loading AppWorld tasks from '{dataset_name}' split...")
    appworld_dataset = AppWorldDataset(dataset_name=dataset_name, max_tasks=max_tasks)
    train_dataset = appworld_dataset.to_dataset()
    print(f"Loaded {len(train_dataset)} tasks")
    
    # Configure training
    training_args = GRPOEnvConfig(
        output_dir=output_dir,
        
        # Basic training settings
        num_train_epochs=1,
        gradient_accumulation_steps=2,
        learning_rate=5e-7,
        warmup_ratio=0.1,
        per_device_train_batch_size=1,  # Lower batch size due to environment execution
        
        # GRPO-specific settings
        num_generations=2,  # Generate 4 code solutions per task
        max_completion_length=256,  # Longer for code generation
        max_prompt_length=2048,  # Longer for task descriptions
        temperature=0.8,  # Higher temperature for diverse code solutions
        
        # Use bfloat16 for efficiency
        bf16=True,
        
        # vLLM configuration
        use_vllm=True,
        vllm_mode="colocate",  # Run vLLM in the same process
        vllm_gpu_memory_utilization=0.3,  # Adjust based on available GPU memory
        vllm_tensor_parallel_size=1,  # Single GPU setup
        
        # Logging
        logging_steps=20,
        save_steps=25,
        log_completions=True,
        
        # Important: keep all columns for reward functions
        remove_unused_columns=False,
        
        # Disable dataset shuffling to maintain task order
        shuffle_dataset=False,
    )
    
    # Set reward weights in training args
    training_args.reward_weights = [0.7, 0.2, 0.1]  # Prioritize task completion
    
    # Create LoRA configuration if enabled
    peft_config = None
    if use_lora:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,  # Apply LoRA to Qwen attention layers
            bias="none",
        )
        print(f"LoRA enabled with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    # Create trainer with AppWorld reward functions
    trainer = GRPOEnvTrainer(
        model=model_path,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[
            AppWorldRewardFunction(max_interactions=max_interactions, dataset_name=dataset_name),  # Main environment reward
            syntax_validity_reward,  # Ensure valid Python syntax
            api_usage_reward,  # Encourage API usage
        ],
        peft_config=peft_config,  # Add LoRA configuration
        dataset_metadata=appworld_dataset,  # Pass the dataset instance to access environments
    )
    
    # Train
    print("Starting training with AppWorld environment feedback...")
    trainer.train()
    
    # Save model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        level=logging.INFO
    )
    
    # Train on AppWorld tasks with LoRA
    train_appworld_model(
        dataset_name="train",
        model_path="model/Qwen/Qwen3-0.6B",
        output_dir="output/qwen-appworld-grpo-lora",
        max_tasks=100,  # Start with a small number of tasks
        max_interactions=30,
        use_lora=True,  # Enable LoRA for efficient training
        lora_r=16,  # LoRA rank
        lora_alpha=32,  # LoRA scaling parameter
        lora_dropout=0.05,  # LoRA dropout
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen model attention layers
    )