#!/usr/bin/env python3
"""
AppWorld Supervised Fine-tuning (SFT) Training Script

This script fine-tunes Qwen3-0.6B on AppWorld conversational trajectory data
for multi-step reasoning and API interaction tasks.
"""

import json
import os
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser
)
from datasets import Dataset
from finetune.trl_main import SFTTrainer, SFTConfig
import transformers


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""
    model_name_or_path: str = field(
        default="model/Qwen/Qwen3-0.6B",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to tokenizer (if different from model)"}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training."""
    data_path: str = field(
        default="appworld_training_dataset.json",
        metadata={"help": "Path to the AppWorld training dataset"}
    )
    max_seq_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length for training"}
    )


@dataclass
class TrainingArguments:
    """Training arguments for AppWorld SFT."""
    output_dir: str = field(
        default="output/qwen3-0.6b-appworld-sft",
        metadata={"help": "Output directory for model and logs"}
    )
    max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length for training"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=2,
        metadata={"help": "Batch size per device during training"}
    )
    gradient_accumulation_steps: int = field(
        default=2,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Initial learning rate"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay for optimization"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing to save memory"}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bf16 mixed precision training"}
    )
    assistant_only_loss: bool = field(
        default=True,
        metadata={"help": "Compute loss only on assistant tokens"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X updates steps"}
    )
    save_steps: int = field(
        default=50,
        metadata={"help": "Save checkpoint every X updates steps"}
    )
    chat_template_path: str = field(
        default="chat_template/qwen_chat_template.jinja",
        metadata={"help": "Path to the chat template file"}
    )


def load_appworld_dataset(data_path: str) -> List[Dict[str, Any]]:
    """Load AppWorld training dataset from JSON file."""
    print(f"Loading AppWorld dataset from: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} training samples")
    
    # Calculate statistics based on conversations field (will be converted to messages later)
    total_conversations = sum(len(sample['conversations']) for sample in data)
    avg_conversations = total_conversations / len(data) if data else 0
    
    print(f"Total conversations: {total_conversations}")
    print(f"Average conversations per sample: {avg_conversations:.1f}")
    
    return data


def format_appworld_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format AppWorld sample for SFTTrainer.
    Converts 'conversations' field to 'messages' field.
    """
    conversations = sample['conversations']
    
    # Convert conversations to messages format
    messages = []
    for conv in conversations:
        role = conv['role']
        content = conv['content']
        
        # Map roles to chat template format
        if role == 'system':
            messages.append({"role": "system", "content": content})
        elif role == 'user':
            messages.append({"role": "user", "content": content})
        elif role == 'assistant':
            messages.append({"role": "assistant", "content": content})
        else:
            # Default to user role for unknown roles
            messages.append({"role": "user", "content": content})
    
    return {
        "messages": messages,
        "task_id": sample.get('task_id', 'unknown')
    }


def preprocess_dataset(data: List[Dict[str, Any]]) -> Dataset:
    """
    Preprocess AppWorld dataset for SFTTrainer.
    """
    print("Preprocessing dataset...")
    
    processed_data = []
    
    for sample in data:
        formatted_sample = format_appworld_sample(sample)
        processed_data.append(formatted_sample)
    
    # Create dataset
    dataset = Dataset.from_list(processed_data)
    
    print(f"Processed {len(dataset)} samples")
    
    return dataset


def setup_model_and_tokenizer(model_args: ModelArguments) -> tuple:
    """Setup model and tokenizer for full parameter fine-tuning."""
    
    print(f"Loading model from: {model_args.model_name_or_path}")
    
    # Load tokenizer
    tokenizer_path = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
    )
    
    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, tokenizer


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set up logging
    transformers.utils.logging.set_verbosity_info()
    
    print("=" * 50)
    print("AppWorld Supervised Fine-tuning")
    print("=" * 50)
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Data: {data_args.data_path}")
    print(f"Output: {training_args.output_dir}")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Load dataset
    raw_data = load_appworld_dataset(data_args.data_path)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    # Preprocess dataset
    dataset = preprocess_dataset(raw_data)
    
    # Use full dataset for training
    train_dataset = dataset
    print(f"Train samples: {len(train_dataset)}")
    
    # Create SFT configuration
    sft_config = SFTConfig(
        max_length=training_args.max_length,
        output_dir=training_args.output_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        gradient_checkpointing=training_args.gradient_checkpointing,
        bf16=training_args.bf16,
        learning_rate=training_args.learning_rate,
        num_train_epochs=training_args.num_train_epochs,
        weight_decay=training_args.weight_decay,
        assistant_only_loss=training_args.assistant_only_loss,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        save_strategy="epoch",
        chat_template_path=training_args.chat_template_path,
        report_to=None,  # Disable wandb/tensorboard for now
        remove_unused_columns=False,
    )
    
    # Initialize SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
    )
    
    # Save training arguments
    config = {
        'model_args': vars(model_args),
        'data_args': vars(data_args),
        'training_args': vars(training_args)
    }
    
    with open(os.path.join(training_args.output_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Starting training...")
    
    # Train
    trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model()
    trainer.save_state()
    
    print(f"Training completed! Model saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()