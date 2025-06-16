# GRPO Training with Conversation Datasets

This guide explains how to use GRPO (Generalized Reward-based Policy Optimization) with conversation datasets.

## Quick Start

```bash
# Simple training with default rewards
python finetune/grpo_conversation_simple.py

# Full example with custom rewards
python finetune/grpo_conversation_trainer.py --dataset HuggingFaceH4/ultrachat_200k
```

## Key Concepts

### 1. Dataset Format
GRPO expects conversation data in this format:
```python
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language..."}
    ]
}
```

### 2. Data Preparation
GRPO needs a `prompt` field containing the conversation history:
```python
def prepare_conversations(example):
    messages = example["messages"]
    # Use all messages except the last assistant response as prompt
    prompt_messages = messages[:-1]  # or up to last user message
    return {"prompt": prompt_messages}
```

### 3. Reward Functions
Reward functions evaluate generated responses. They receive:
- `completions`: List of generated responses
- `prompts`: Original prompts (optional)
- `**kwargs`: Additional data from dataset

```python
def my_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        response = completion[0]["content"]
        # Calculate reward (0.0 to 1.0)
        reward = calculate_quality(response)
        rewards.append(reward)
    return rewards
```

## Common Reward Functions for Conversations

### Length Control
```python
def length_reward(completions, **kwargs):
    """Prefer responses between 50-200 words."""
    rewards = []
    for completion in completions:
        words = len(completion[0]["content"].split())
        if 50 <= words <= 200:
            reward = 1.0
        else:
            reward = 0.5
        rewards.append(reward)
    return rewards
```

### Helpfulness
```python
def helpfulness_reward(completions, **kwargs):
    """Reward helpful language patterns."""
    helpful_phrases = ["I can help", "Here's how", "Let me explain"]
    rewards = []
    for completion in completions:
        text = completion[0]["content"].lower()
        score = sum(0.3 for phrase in helpful_phrases if phrase.lower() in text)
        rewards.append(min(1.0, score))
    return rewards
```

### Safety
```python
def safety_reward(completions, **kwargs):
    """Penalize unsafe content."""
    unsafe_words = ["dangerous", "illegal", "harmful"]
    rewards = []
    for completion in completions:
        text = completion[0]["content"].lower()
        has_unsafe = any(word in text for word in unsafe_words)
        rewards.append(0.0 if has_unsafe else 1.0)
    return rewards
```

### Coherence with Context
```python
def coherence_reward(completions, prompts, **kwargs):
    """Check if response is relevant to the conversation."""
    rewards = []
    for completion, prompt in zip(completions, prompts):
        response = completion[0]["content"]
        # Get last user message
        last_user_msg = ""
        for msg in reversed(prompt):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break
        
        # Simple relevance check (you can use embeddings for better results)
        user_words = set(last_user_msg.lower().split())
        response_words = set(response.lower().split())
        overlap = len(user_words & response_words)
        
        rewards.append(min(1.0, overlap * 0.2))
    return rewards
```

## Training Configuration

### Basic Settings
```python
training_args = GRPOConfig(
    output_dir="output/my-conversation-model",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,
    bf16=True,  # Use bfloat16 for efficiency
)
```

### GRPO-Specific Settings
```python
training_args = GRPOConfig(
    # How many responses to generate per prompt
    num_generations=4,
    
    # Maximum length of generated responses
    max_completion_length=256,
    
    # Maximum length of conversation history
    max_prompt_length=512,
    
    # Sampling temperature (higher = more diverse)
    temperature=0.7,
    
    # Keep all dataset columns for reward functions
    remove_unused_columns=False,
)
```

## Complete Example

```python
from datasets import load_dataset
from trl_main import GRPOConfig, GRPOTrainer

# Load dataset
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train[:1000]")

# Prepare data
def prepare_data(example):
    messages = example["messages"]
    prompt = messages[:-1]  # All except last assistant response
    return {"prompt": prompt}

dataset = dataset.map(prepare_data)

# Define rewards
def quality_reward(completions, **kwargs):
    # Your reward logic here
    return [1.0] * len(completions)

# Configure training
args = GRPOConfig(
    output_dir="output/my-model",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    num_generations=4,
    max_completion_length=200,
    learning_rate=5e-7,
    bf16=True,
)

# Train
trainer = GRPOTrainer(
    model="model/Qwen/Qwen3-0.6B",
    reward_funcs=[quality_reward],
    args=args,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model()
```

## Tips for Better Results

1. **Combine Multiple Rewards**: Use 3-5 complementary reward functions
2. **Balance Rewards**: Weight different objectives appropriately
3. **Start Small**: Test with small datasets first (1-5% splits)
4. **Monitor Training**: Use tensorboard to track reward scores
5. **Iterate on Rewards**: Refine reward functions based on generated samples

## Common Issues

### Out of Memory
- Reduce `per_device_train_batch_size`
- Reduce `num_generations`
- Use gradient checkpointing
- Use LoRA for parameter-efficient training

### Poor Quality Responses
- Check reward function logic
- Increase `num_generations` for more diverse samples
- Adjust `temperature` for better exploration
- Use more training data

### Slow Training
- Use `bf16=True` for faster computation
- Reduce `max_completion_length`
- Use smaller models for prototyping
- Enable Flash Attention if available