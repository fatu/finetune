# train_grpo.py
from datasets import load_dataset
from peft import LoraConfig
from trl_main import GRPOConfig, GRPOTrainer
import torch

# Load the dataset
dataset = load_dataset("data/trl-lib/tldr", split="train")

training_args = GRPOConfig(
    output_dir="output/Qwen2-0.5B-GRPO",
    learning_rate=1e-4,
    logging_steps=10,
    num_generations=4,
    gradient_accumulation_steps=2,
    max_completion_length=12,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    model_init_kwargs={"torch_dtype": torch.bfloat16},
    num_train_epochs=1,
    save_steps=1000,
    # # DeepSpeed configuration
    # deepspeed="finetune/ds_zero2_no_offload.json",
    # fp16=True,  # Enable FP16 as configured in DeepSpeed
)
trainer = GRPOTrainer(
    model="model/Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs="model/weqweasdas/RM-Gemma-2B",
    args=training_args,
    train_dataset=dataset,
    # peft_config=LoraConfig(
    #     task_type="CAUSAL_LM",
    #     r=8,
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen model attention layers
    # ),
)

trainer.train()
trainer.save_model(training_args.output_dir)