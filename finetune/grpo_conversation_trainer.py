# grpo_conversation_trainer.py
"""
GRPO trainer for conversation datasets with customizable reward functions.

This script demonstrates how to use GRPO (Generalized Reward-based Policy Optimization)
with conversation datasets. It includes several reward functions suitable for dialogue:
- Response quality (coherence, relevance)
- Safety and helpfulness
- Format compliance
- Custom reward functions
"""

from datasets import load_dataset
from peft import LoraConfig
from trl_main import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer
import torch
import re
from typing import List, Dict, Any, Optional


# Example conversation dataset format:
# Each example should have a "messages" field with a list of conversation turns
# [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello!"},
#     {"role": "assistant", "content": "Hi! How can I help you today?"}
# ]


def prepare_conversation_dataset(dataset_name: str, split: str = "train[:1%]"):
    """
    Load and prepare a conversation dataset for GRPO training.
    
    Args:
        dataset_name: Name of the dataset (e.g., "HuggingFaceH4/ultrachat_200k")
        split: Dataset split to use
    
    Returns:
        Prepared dataset with "prompt" field containing conversation history
    """
    dataset = load_dataset(dataset_name, split=split)
    
    def format_conversation(example):
        """Convert messages to prompt format for GRPO."""
        # For GRPO, we need to separate the prompt (conversation history) 
        # from what we want to generate (the last assistant response)
        
        if "messages" in example:
            messages = example["messages"]
        elif "conversations" in example:
            messages = example["conversations"]
        else:
            raise ValueError("Dataset must have 'messages' or 'conversations' field")
        
        # Find the last user message and prepare prompt up to that point
        prompt_messages = []
        for i, msg in enumerate(messages):
            if msg["role"] == "user" and i < len(messages) - 1:
                # Include all messages up to and including this user message
                prompt_messages = messages[:i+1]
        
        return {"prompt": prompt_messages}
    
    return dataset.map(format_conversation)


# Reward Functions for Conversations

def response_length_reward(completions: List[List[Dict]], **kwargs) -> List[float]:
    """
    Reward function that encourages responses of appropriate length.
    Too short: low reward, too long: penalty
    """
    rewards = []
    for completion in completions:
        if completion and len(completion) > 0:
            response = completion[0].get("content", "")
            word_count = len(response.split())
            
            # Ideal response length: 20-150 words
            if 20 <= word_count <= 150:
                reward = 1.0
            elif word_count < 20:
                reward = word_count / 20  # Linear penalty for too short
            else:
                # Penalty for too long responses
                reward = max(0.0, 1.0 - (word_count - 150) / 200)
            
            rewards.append(reward)
        else:
            rewards.append(0.0)
    
    return rewards


def helpfulness_reward(completions: List[List[Dict]], **kwargs) -> List[float]:
    """
    Reward function that checks for helpful patterns in responses.
    """
    helpful_patterns = [
        r"I can help",
        r"Here's how",
        r"Let me explain",
        r"I'd be happy to",
        r"Sure,",
        r"Of course",
        r"Here are some",
        r"To answer your question",
    ]
    
    rewards = []
    for completion in completions:
        if completion and len(completion) > 0:
            response = completion[0].get("content", "").lower()
            
            # Check for helpful patterns
            score = 0.0
            for pattern in helpful_patterns:
                if re.search(pattern.lower(), response):
                    score += 0.3
            
            # Cap at 1.0
            rewards.append(min(1.0, score))
        else:
            rewards.append(0.0)
    
    return rewards


def safety_reward(completions: List[List[Dict]], **kwargs) -> List[float]:
    """
    Reward function that penalizes unsafe or inappropriate content.
    """
    unsafe_patterns = [
        r"I cannot",
        r"I'm not able to",
        r"inappropriate",
        r"harmful",
        r"dangerous",
        r"illegal",
        r"unethical",
    ]
    
    rewards = []
    for completion in completions:
        if completion and len(completion) > 0:
            response = completion[0].get("content", "").lower()
            
            # Check for refusal patterns (which might indicate unsafe request)
            has_refusal = any(re.search(pattern, response) for pattern in unsafe_patterns[:2])
            
            # If refusing, that's actually good (assuming unsafe request)
            if has_refusal:
                reward = 0.8
            else:
                # Check if response contains unsafe content
                has_unsafe = any(re.search(pattern, response) for pattern in unsafe_patterns[2:])
                reward = 0.0 if has_unsafe else 1.0
            
            rewards.append(reward)
        else:
            rewards.append(0.0)
    
    return rewards


def coherence_reward(completions: List[List[Dict]], prompts: List[List[Dict]], **kwargs) -> List[float]:
    """
    Reward function that checks if response is coherent with the conversation context.
    This is a simple implementation - you might want to use a more sophisticated approach.
    """
    rewards = []
    
    for completion, prompt in zip(completions, prompts):
        if completion and len(completion) > 0 and prompt:
            response = completion[0].get("content", "").lower()
            
            # Get the last user message
            last_user_msg = ""
            for msg in reversed(prompt):
                if msg.get("role") == "user":
                    last_user_msg = msg.get("content", "").lower()
                    break
            
            # Simple coherence check: response should reference something from user message
            # This is very basic - consider using embeddings or a classifier
            score = 0.0
            
            # Check for topic relevance (shared words)
            user_words = set(last_user_msg.split())
            response_words = set(response.split())
            common_words = user_words.intersection(response_words)
            
            # Remove common stop words
            stop_words = {"the", "a", "an", "is", "are", "was", "were", "i", "you", "he", "she", "it", "we", "they"}
            common_words = common_words - stop_words
            
            if len(common_words) > 0:
                score = min(1.0, len(common_words) * 0.2)
            
            rewards.append(score)
        else:
            rewards.append(0.0)
    
    return rewards


def format_compliance_reward(completions: List[List[Dict]], **kwargs) -> List[float]:
    """
    Reward function for specific format requirements.
    Customize this based on your needs.
    """
    rewards = []
    
    for completion in completions:
        if completion and len(completion) > 0:
            response = completion[0].get("content", "")
            
            # Example: Check if response follows certain format
            # - Starts with capital letter
            # - Ends with proper punctuation
            # - No excessive caps
            
            score = 1.0
            
            # Check capitalization
            if response and not response[0].isupper():
                score -= 0.2
            
            # Check ending punctuation
            if response and response[-1] not in ".!?":
                score -= 0.2
            
            # Check for excessive caps (shouting)
            caps_ratio = sum(1 for c in response if c.isupper()) / max(len(response), 1)
            if caps_ratio > 0.3:
                score -= 0.3
            
            rewards.append(max(0.0, score))
        else:
            rewards.append(0.0)
    
    return rewards


# Custom reward function that combines multiple signals
def combined_conversation_reward(completions: List[List[Dict]], prompts: List[List[Dict]], **kwargs) -> List[float]:
    """
    Combined reward function that balances multiple objectives.
    """
    # Get individual rewards
    length_rewards = response_length_reward(completions, **kwargs)
    helpful_rewards = helpfulness_reward(completions, **kwargs)
    safety_rewards = safety_reward(completions, **kwargs)
    coherence_rewards = coherence_reward(completions, prompts=prompts, **kwargs)
    format_rewards = format_compliance_reward(completions, **kwargs)
    
    # Combine with weights
    combined_rewards = []
    for i in range(len(completions)):
        reward = (
            0.2 * length_rewards[i] +
            0.3 * helpful_rewards[i] +
            0.2 * safety_rewards[i] +
            0.2 * coherence_rewards[i] +
            0.1 * format_rewards[i]
        )
        combined_rewards.append(reward)
    
    return combined_rewards


def main():
    # Example usage with a conversation dataset
    
    # Load your conversation dataset
    # Examples of conversation datasets:
    # - "HuggingFaceH4/ultrachat_200k"
    # - "berkeley-nest/Nectar"
    # - "lmsys/chatbot_arena_conversations"
    # - Your custom dataset
    
    dataset_name = "HuggingFaceH4/ultrachat_200k"  # Change this to your dataset
    
    # Prepare dataset
    train_dataset = prepare_conversation_dataset(dataset_name, split="train[:5%]")
    eval_dataset = prepare_conversation_dataset(dataset_name, split="test[:1%]")
    
    # Configure LoRA for efficient training
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    # Configure GRPO training
    training_args = GRPOConfig(
        output_dir="output/conversation-grpo-model",
        learning_rate=5e-7,
        num_train_epochs=1,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        bf16=True,
        
        # GRPO-specific parameters
        num_generations=4,  # Number of completions to generate per prompt
        max_completion_length=256,  # Max tokens for generated responses
        max_prompt_length=512,  # Max tokens for conversation history
        temperature=0.8,  # Sampling temperature
        
        # Logging and saving
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=50,
        report_to=["tensorboard"],
        
        # Important for conversation datasets
        remove_unused_columns=False,  # Keep all columns for reward functions
    )
    
    # Initialize trainer with multiple reward functions
    trainer = GRPOTrainer(
        model="model/Qwen/Qwen3-0.6B",  # Change to your model
        reward_funcs=[
            response_length_reward,
            helpfulness_reward,
            safety_reward,
            coherence_reward,
            combined_conversation_reward,  # You can use individual or combined
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model(training_args.output_dir)
    
    print(f"Training completed! Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    # You can also run with command line arguments
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--model", type=str, default="model/Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default="output/conversation-grpo")
    parser.add_argument("--train_split", type=str, default="train[:5%]")
    parser.add_argument("--eval_split", type=str, default="test[:1%]")
    
    args = parser.parse_args()
    
    # Run with custom arguments
    train_dataset = prepare_conversation_dataset(args.dataset, split=args.train_split)
    eval_dataset = prepare_conversation_dataset(args.dataset, split=args.eval_split)
    
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=5e-7,
        num_train_epochs=1,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=2,
        num_generations=4,
        max_completion_length=256,
        max_prompt_length=512,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        remove_unused_columns=False,
    )
    
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[combined_conversation_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)