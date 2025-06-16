# train_grpo.py
from datasets import load_dataset
from peft import LoraConfig
from trl_main import GRPOConfig, GRPOTrainer
import torch
import re
from math_verify import LatexExtractionConfig, parse, verify

# Load the dataset
train_dataset, test_dataset = load_dataset("AI-MO/NuminaMath-TIR", split=["train[:5%]", "test[:5%]"])

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

train_dataset = train_dataset.map(make_conversation)
test_dataset = test_dataset.map(make_conversation)

train_dataset = train_dataset.remove_columns(["messages", "problem"])

# reward functions
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs["solution"]
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir="output/Qwen3-0.6B-GRPO-test",
    learning_rate=1e-5,
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    bf16=True,
    # Parameters that control de data preprocessing
    max_completion_length=64,  # default: 256
    num_generations=4,  # default: 8
    max_prompt_length=128,  # default: 512
    # Parameters related to reporting and saving
    report_to=["tensorboard"],
    logging_steps=10,
    save_strategy="steps",
    save_steps=10,
)

trainer = GRPOTrainer(
    model="model/Qwen/Qwen3-0.6B", 
    reward_funcs=[format_reward, accuracy_reward], 
    args=training_args, 
    train_dataset=train_dataset
)

trainer.train()
trainer.save_model(training_args.output_dir)