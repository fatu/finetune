import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from trl import GKDTrainer, GKDConfig
import re
from tqdm import tqdm

# Step 1: Load and Prepare Dataset
dataset = load_dataset("microsoft/orca-math-word-problems-200k")

def format_as_messages(example):
    return {
        "messages": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]}
        ]
    }

formatted_dataset = dataset["train"].map(format_as_messages)
# Use a subset for demonstration; in practice, use full dataset
train_dataset = formatted_dataset.select(range(10000))  # Example subset
eval_dataset = formatted_dataset.select(range(10000, 10500))  # Small eval set

# Step 2: Load Teacher and Student Models
teacher_model_name = "Qwen/Qwen3-7B"  # Large teacher
student_model_name = "Qwen/Qwen3-1.7B"   # Smaller student
tokenizer = AutoTokenizer.from_pretrained(student_model_name)
student_model = AutoModelForCausalLM.from_pretrained(student_model_name)
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)

# Step 3: Configure and Run Distillation
gkd_config = GKDConfig(
    output_dir="./distilled_qwen3_math",
    per_device_train_batch_size=2,  # Adjust based on GPU
    per_device_eval_batch_size=2,
    num_train_epochs=1,  # Increase for better results
    learning_rate=5e-5,
    beta=0.0,  # Approximates forward KL divergence
    lmbda=0.5,  # Balance between on-policy and teacher data
    temperature=0.9,
    max_new_tokens=256,
    fp16=True,  # Enable if using GPU
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
)

trainer = GKDTrainer(
    model=student_model,
    teacher_model=teacher_model,
    args=gkd_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Save the distilled student model
trainer.save_model("./distilled_qwen3_math_final")

# Step 4: Evaluation on AIME 2025
# Load AIME dataset
aime_dataset = load_dataset("opencompass/AIME2025")["train"]