# grpo_conversation_simple.py
"""
Simple example of using GRPO with conversation datasets.
This script shows the minimal setup needed to train with conversation data.
"""

from datasets import load_dataset
from trl_main import GRPOConfig, GRPOTrainer
import re


def prepare_conversations(example):
    """
    Convert conversation format to GRPO format.
    GRPO expects a 'prompt' field containing the conversation history.
    """
    # Handle different dataset formats
    if "messages" in example:
        messages = example["messages"]
    elif "conversations" in example:
        messages = example["conversations"]
    else:
        # If your dataset has a different format, adjust here
        return example
    
    # For GRPO, we want to predict the last assistant response
    # So the prompt should be everything except the last assistant message
    prompt_messages = []
    
    # Find the last assistant message
    last_assistant_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_assistant_idx = i
            break
    
    # Include all messages up to (but not including) the last assistant response
    if last_assistant_idx > 0:
        prompt_messages = messages[:last_assistant_idx]
    else:
        # If no assistant message or it's the first message, use all but last
        prompt_messages = messages[:-1] if len(messages) > 1 else messages
    
    return {"prompt": prompt_messages}


# Simple reward functions

def length_preference_reward(completions, **kwargs):
    """Prefer medium-length responses (50-200 words)."""
    rewards = []
    for completion in completions:
        if completion and len(completion) > 0:
            text = completion[0]["content"]
            word_count = len(text.split())
            
            if 50 <= word_count <= 200:
                reward = 1.0
            elif word_count < 50:
                reward = word_count / 50
            else:
                reward = max(0.1, 1.0 - (word_count - 200) / 300)
            
            rewards.append(reward)
        else:
            rewards.append(0.0)
    return rewards


def politeness_reward(completions, **kwargs):
    """Reward polite and helpful language."""
    polite_phrases = [
        "please", "thank you", "you're welcome", "happy to help",
        "glad to", "of course", "certainly", "I'd be happy to",
        "let me help", "here's", "I can help"
    ]
    
    rewards = []
    for completion in completions:
        if completion and len(completion) > 0:
            text = completion[0]["content"].lower()
            
            # Count polite phrases
            politeness_score = sum(
                0.3 for phrase in polite_phrases 
                if phrase in text
            )
            
            rewards.append(min(1.0, politeness_score))
        else:
            rewards.append(0.0)
    return rewards


def no_repetition_reward(completions, **kwargs):
    """Penalize repetitive responses."""
    rewards = []
    for completion in completions:
        if completion and len(completion) > 0:
            text = completion[0]["content"].lower()
            words = text.split()
            
            if len(words) == 0:
                rewards.append(0.0)
                continue
            
            # Check for repeated phrases (3+ words)
            repeated = False
            for i in range(len(words) - 5):
                phrase = " ".join(words[i:i+3])
                if text.count(phrase) > 1:
                    repeated = True
                    break
            
            rewards.append(0.5 if repeated else 1.0)
        else:
            rewards.append(0.0)
    return rewards


# Main training script
def train_conversation_model(
    dataset_name="neuralmagic/ultrachat_2k",
    model_path="model/Qwen/Qwen3-0.6B",
    output_dir="output/conversation-grpo",
    train_samples=1000,
):
    """
    Train a model on conversation data using GRPO.
    
    Args:
        dataset_name: HuggingFace dataset with conversations
        model_path: Path to base model
        output_dir: Where to save the trained model
        train_samples: Number of training samples to use
    """
    
    # Load and prepare dataset
    print(f"Loading dataset: {dataset_name}")
    train_dataset = load_dataset(dataset_name, split=f"train_sft[:{train_samples}]")
    
    # Convert to GRPO format
    train_dataset = train_dataset.map(prepare_conversations)
    train_dataset = train_dataset.remove_columns(["messages"])
    
    # Configure training
    training_args = GRPOConfig(
        output_dir=output_dir,
        
        # Basic training settings
        num_train_epochs=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,
        warmup_ratio=0.1,
        per_device_train_batch_size=2,
        
        # GRPO-specific settings
        num_generations=4,  # Generate 4 responses per prompt
        max_completion_length=64,  # Max response length
        max_prompt_length=512,  # Max conversation history length
        temperature=0.7,  # Sampling temperature
        
        # Use bfloat16 for efficiency
        bf16=True,
        
        # Logging
        logging_steps=10,
        save_steps=50,
        
        # Important: keep all columns for reward functions
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model_path,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[
            length_preference_reward,
            politeness_reward,
            no_repetition_reward,
        ],
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    # Example 1: Train on UltraChat dataset
    train_conversation_model(
        dataset_name="neuralmagic/ultrachat_2k",
        model_path="model/Qwen/Qwen3-0.6B",
        output_dir="output/qwen-ultrachat-grpo",
        train_samples=2000
    )
    
    # Example 2: Train on your own dataset
    # Your dataset should have a 'messages' or 'conversations' field
    # with format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    
    # train_conversation_model(
    #     dataset_name="your-username/your-conversation-dataset",
    #     model_path="model/Qwen/Qwen3-0.6B",
    #     output_dir="output/custom-conversation-grpo",
    #     train_samples=5000,
    #     eval_samples=500
    # )