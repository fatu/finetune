import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
import json
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class DistillationConfig:
    # model params
    teacher_model: str = "Qwen/Qwen3-1.7B"
    student_model: str = "Qwen/Qwen3-0.6B"
    
    # training params
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # distillation params
    temperature: float = 3.0
    alpha: float = 0.7  # weight for distillation loss
    beta: float = 0.3   # weight for student loss
    
    # others
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50
    output_dir: str = "./outputs"
    max_length: int = 2048
    seed: int = 42
    
    # data
    dataset_name: str = "trl-lib/tldr"
    data_path: Optional[str] = None
    
    # device
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    use_flash_attention: bool = True
    
    # memory optimization
    gradient_checkpointing: bool = True
    teacher_cpu_offload: bool = False

class DistillationTrainer:
    """Distillation Trainer Class"""
    def __init__(
        self,
        config: DistillationConfig,
        teacher_model: Optional[Union[str, PreTrainedModel]] = None,
        student_model: Optional[Union[str, PreTrainedModel]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        accelerator: Optional[Accelerator] = None,
    ):
        self.config = config
        
        # Initialize accelerator
        if accelerator is None:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                mixed_precision="bf16" if config.torch_dtype == "bfloat16" else "fp16",
                project_dir=config.output_dir,
            )
        else:
            self.accelerator = accelerator
            
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Initialize models and tokenizer
        self._init_models_and_tokenizer(teacher_model, student_model, tokenizer)
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.lr_scheduler = None
        
        # Initialize training state
        self.global_step = 0
        self.current_epoch = 0
        
    def _init_models_and_tokenizer(
        self, 
        teacher_model: Optional[Union[str, PreTrainedModel]],
        student_model: Optional[Union[str, PreTrainedModel]], 
        tokenizer: Optional[PreTrainedTokenizerBase]
    ):
        """Initialize teacher model, student model, and tokenizer."""
        # Convert torch_dtype string to actual dtype
        torch_dtype = getattr(torch, self.config.torch_dtype, torch.float32)
        
        # Model initialization kwargs
        model_init_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        # Don't use device_map with Flash Attention to avoid placement issues
        # We'll manually move models to GPU after loading
        
        if self.config.use_flash_attention:
            model_init_kwargs["attn_implementation"] = "flash_attention_2"
            
        # Load teacher model
        if teacher_model is None:
            teacher_model = self.config.teacher_model
        if isinstance(teacher_model, str):
            teacher_kwargs = model_init_kwargs.copy()
            if self.config.teacher_cpu_offload:
                logger.info(f"Loading teacher model on CPU: {teacher_model}")
                teacher_kwargs["device_map"] = "cpu"
                # Remove flash attention for CPU model
                teacher_kwargs.pop("attn_implementation", None)
            else:
                logger.info(f"Loading teacher model: {teacher_model}")
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model,
                **teacher_kwargs
            )
        else:
            self.teacher_model = teacher_model
            
        # Move teacher model to GPU if not using CPU offloading
        if not self.config.teacher_cpu_offload:
            self.teacher_model = self.teacher_model.to(self.accelerator.device)
            if torch.cuda.is_available():
                logger.info(f"GPU memory after loading teacher: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
        # Teacher model in eval mode (no gradients needed)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        # Load student model
        if student_model is None:
            student_model = self.config.student_model
        if isinstance(student_model, str):
            logger.info(f"Loading student model: {student_model}")
            self.student_model = AutoModelForCausalLM.from_pretrained(
                student_model,
                **model_init_kwargs
            )
        else:
            self.student_model = student_model
            
        # Move student model to GPU
        self.student_model = self.student_model.to(self.accelerator.device)
        if torch.cuda.is_available():
            logger.info(f"GPU memory after loading student: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Student model in train mode
        self.student_model.train()
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            self.student_model.gradient_checkpointing_enable()
            # Disable cache when using gradient checkpointing
            if hasattr(self.student_model.config, 'use_cache'):
                self.student_model.config.use_cache = False
        
        # Load tokenizer
        if tokenizer is None:
            logger.info(f"Loading tokenizer from: {self.config.student_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.student_model if isinstance(student_model, str) else self.config.student_model,
                trust_remote_code=True,
                padding_side="left",
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer
            
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute distillation loss using KL divergence between teacher and student outputs."""
        if temperature is None:
            temperature = self.config.temperature
            
        # Shift logits and labels for language modeling
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        shift_student_logits = shift_student_logits.view(-1, shift_student_logits.size(-1))
        shift_teacher_logits = shift_teacher_logits.view(-1, shift_teacher_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Create mask for valid tokens (not padding)
        mask = shift_labels != self.tokenizer.pad_token_id
        
        # Student loss (cross entropy with ground truth)
        student_loss = F.cross_entropy(
            shift_student_logits[mask],
            shift_labels[mask],
            reduction='mean'
        )
        
        # More memory-efficient distillation loss computation
        # Compute in chunks to avoid memory explosion
        masked_student_logits = shift_student_logits[mask]
        masked_teacher_logits = shift_teacher_logits[mask]
        
        # Use smaller chunks if tensor is too large
        chunk_size = min(1000, masked_student_logits.size(0))
        distill_loss = 0.0
        num_chunks = 0
        
        for i in range(0, masked_student_logits.size(0), chunk_size):
            end_idx = min(i + chunk_size, masked_student_logits.size(0))
            
            student_chunk = masked_student_logits[i:end_idx]
            teacher_chunk = masked_teacher_logits[i:end_idx]
            
            student_log_probs = F.log_softmax(student_chunk / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_chunk / temperature, dim=-1)
            
            chunk_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean'
            )
            
            distill_loss += chunk_loss
            num_chunks += 1
            
            # Clear intermediate tensors
            del student_log_probs, teacher_probs, chunk_loss
            
        distill_loss = (distill_loss / num_chunks) * (temperature ** 2)
        
        # Combined loss
        total_loss = self.config.alpha * distill_loss + self.config.beta * student_loss
        
        # Clear memory
        del masked_student_logits, masked_teacher_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "loss": total_loss,
            "distill_loss": distill_loss,
            "student_loss": student_loss,
        }
        
    @torch.no_grad()
    def get_teacher_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get teacher model predictions without gradient computation."""
        # Move teacher to GPU for inference if it's on CPU
        teacher_device = next(self.teacher_model.parameters()).device
        student_device = input_ids.device
        
        if teacher_device != student_device:
            # Move inputs to teacher device
            teacher_input_ids = input_ids.to(teacher_device)
            teacher_attention_mask = attention_mask.to(teacher_device)
        else:
            teacher_input_ids = input_ids
            teacher_attention_mask = attention_mask
            
        outputs = self.teacher_model(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask,
            use_cache=False,
        )
        
        # Move logits back to student device and clear cache
        if teacher_device != student_device:
            logits = outputs.logits.to(student_device)
        else:
            logits = outputs.logits
            
        # Clear CUDA cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return logits
        
    def prepare_dataset(self) -> Dataset:
        """Load and prepare the dataset for training."""
        if self.config.data_path and os.path.exists(self.config.data_path):
            # Load from local file
            with open(self.config.data_path, 'r') as f:
                data = json.load(f)
            dataset = Dataset.from_dict({"text": data})
        else:
            # Load from HuggingFace datasets
            dataset = load_dataset(self.config.dataset_name, split="train")
            
        # Tokenize the dataset
        def tokenize_function(examples):
            # Handle different dataset formats
            if "text" in examples:
                texts = examples["text"]
            elif "prompt" in examples and "completion" in examples:
                # Handle trl-lib/tldr format and similar datasets
                texts = [f"{p}{c}" for p, c in zip(examples["prompt"], examples["completion"])]
            elif "question" in examples and "answer" in examples:
                texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples["question"], examples["answer"])]
            else:
                raise ValueError("Unsupported dataset format")
                
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.config.max_length,
            )
            
            # Keep as lists - PyTorch DataLoader will handle tensor conversion
            
            return tokenized
            
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        return tokenized_dataset
        
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Create optimizer and learning rate scheduler."""
        # Get parameters that require gradients
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.student_model.named_parameters() if p.requires_grad],
                "weight_decay": self.config.weight_decay,
            }
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
        )
        
        # Create scheduler
        num_warmup_steps = int(self.config.warmup_ratio * num_training_steps)
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
    def train(self):
        """Main training loop."""
        # Prepare dataset
        train_dataset = self.prepare_dataset()
        
        # Create dataloader
        def collate_fn(batch):
            """Custom collate function to handle tokenized data."""
            input_ids = torch.tensor([item["input_ids"] for item in batch])
            attention_mask = torch.tensor([item["attention_mask"] for item in batch])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        # Calculate total training steps
        num_update_steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
        total_training_steps = self.config.num_epochs * num_update_steps_per_epoch
        
        # Create optimizer and scheduler
        self.create_optimizer_and_scheduler(total_training_steps)
        
        # Prepare for distributed training
        self.student_model, self.optimizer, train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.student_model, self.optimizer, train_dataloader, self.lr_scheduler
        )
        
        # Move teacher model to device only if not using CPU offloading
        if not self.config.teacher_cpu_offload:
            self.teacher_model = self.teacher_model.to(self.accelerator.device)
            
        # Initialize tensorboard
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("distillation_training")
            
        # Training loop
        progress_bar = tqdm(
            range(total_training_steps),
            disable=not self.accelerator.is_local_main_process,
            desc="Training"
        )
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0
            epoch_distill_loss = 0
            epoch_student_loss = 0
            
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.student_model):
                    # Get inputs
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = input_ids.clone()
                    
                    # Get teacher logits (no gradient)
                    teacher_logits = self.get_teacher_logits(input_ids, attention_mask)
                    
                    # Get student logits
                    student_outputs = self.student_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    student_logits = student_outputs.logits
                    
                    # Compute losses
                    loss_dict = self.compute_distillation_loss(
                        student_logits=student_logits,
                        teacher_logits=teacher_logits,
                        labels=labels,
                    )
                    
                    loss = loss_dict["loss"]
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(
                            self.student_model.parameters(),
                            self.config.max_grad_norm
                        )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Update metrics
                    epoch_loss += loss.detach().float()
                    epoch_distill_loss += loss_dict["distill_loss"].detach().float()
                    epoch_student_loss += loss_dict["student_loss"].detach().float()
                    
                # Update progress bar
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = self.accelerator.gather(epoch_loss).mean().item() / (step + 1)
                        avg_distill_loss = self.accelerator.gather(epoch_distill_loss).mean().item() / (step + 1)
                        avg_student_loss = self.accelerator.gather(epoch_student_loss).mean().item() / (step + 1)
                        
                        self.accelerator.log(
                            {
                                "train/loss": avg_loss,
                                "train/distill_loss": avg_distill_loss,
                                "train/student_loss": avg_student_loss,
                                "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                                "train/epoch": epoch,
                            },
                            step=self.global_step,
                        )
                        
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "distill": f"{avg_distill_loss:.4f}",
                            "student": f"{avg_student_loss:.4f}",
                        })
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()
                        
            # End of epoch logging
            avg_epoch_loss = epoch_loss.item() / len(train_dataloader)
            logger.info(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
            
        # Save final checkpoint
        self.save_checkpoint(is_final=True)
        
        # End training
        self.accelerator.end_training()
        progress_bar.close()
        
    def save_checkpoint(self, is_final: bool = False):
        """Save model checkpoint."""
        if self.accelerator.is_main_process:
            checkpoint_name = "final_checkpoint" if is_final else f"checkpoint-{self.global_step}"
            checkpoint_path = os.path.join(self.config.output_dir, checkpoint_name)
            
            # Create checkpoint directory
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Save model
            unwrapped_model = self.accelerator.unwrap_model(self.student_model)
            unwrapped_model.save_pretrained(
                checkpoint_path,
                save_function=self.accelerator.save,
                state_dict=self.accelerator.get_state_dict(self.student_model),
            )
            
            # Save tokenizer
            self.tokenizer.save_pretrained(checkpoint_path)
            
            # Save training state
            training_state = {
                "global_step": self.global_step,
                "epoch": self.current_epoch,
                "config": self.config.__dict__,
            }
            
            torch.save(
                training_state,
                os.path.join(checkpoint_path, "training_state.pt")
            )
            
            logger.info(f"Checkpoint saved at {checkpoint_path}")
            
    def evaluate(self, eval_dataset: Optional[Dataset] = None):
        """Evaluate the student model."""
        if eval_dataset is None:
            eval_dataset = self.prepare_dataset()
            
        def collate_fn(batch):
            """Custom collate function to handle tokenized data."""
            input_ids = torch.tensor([item["input_ids"] for item in batch])
            attention_mask = torch.tensor([item["attention_mask"] for item in batch])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        eval_dataloader = self.accelerator.prepare(eval_dataloader)
        
        self.student_model.eval()
        total_eval_loss = 0
        total_eval_perplexity = 0
        eval_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not self.accelerator.is_local_main_process):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = input_ids.clone()
                
                outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                loss = outputs.loss
                total_eval_loss += self.accelerator.gather(loss).mean().item()
                
                # Calculate perplexity
                perplexity = torch.exp(loss)
                total_eval_perplexity += self.accelerator.gather(perplexity).mean().item()
                
                eval_steps += 1
                
        avg_eval_loss = total_eval_loss / eval_steps
        avg_eval_perplexity = total_eval_perplexity / eval_steps
        
        self.student_model.train()
        
        return {
            "eval_loss": avg_eval_loss,
            "eval_perplexity": avg_eval_perplexity,
        }
        
    @classmethod
    def from_config_file(cls, config_path: str, **kwargs):
        """Create trainer from configuration file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            
        # Merge config from file with kwargs
        for key, value in kwargs.items():
            if hasattr(DistillationConfig, key):
                config_dict[key] = value
                
        config = DistillationConfig(**config_dict)
        return cls(config)


def main():
    """Main function to run distillation training."""
    import argparse
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    parser = argparse.ArgumentParser(description="Knowledge Distillation Trainer")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--teacher_model", type=str, help="Teacher model name or path")
    parser.add_argument("--student_model", type=str, help="Student model name or path")
    parser.add_argument("--dataset_name", type=str, help="Dataset name from HuggingFace")
    parser.add_argument("--output_dir", type=str, help="Output directory for checkpoints")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--temperature", type=float, help="Distillation temperature")
    parser.add_argument("--alpha", type=float, help="Weight for distillation loss")
    parser.add_argument("--beta", type=float, help="Weight for student loss")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        trainer = DistillationTrainer.from_config_file(
            args.config,
            teacher_model=args.teacher_model or None,
            student_model=args.student_model or None,
            dataset_name=args.dataset_name or None,
            output_dir=args.output_dir or None,
            num_epochs=args.num_epochs or None,
            batch_size=args.batch_size or None,
            learning_rate=args.learning_rate or None,
            temperature=args.temperature or None,
            alpha=args.alpha or None,
            beta=args.beta or None,
        )
    else:
        # Create config from arguments
        config_dict = {}
        if args.teacher_model:
            config_dict["teacher_model"] = args.teacher_model
        if args.student_model:
            config_dict["student_model"] = args.student_model
        if args.dataset_name:
            config_dict["dataset_name"] = args.dataset_name
        if args.output_dir:
            config_dict["output_dir"] = args.output_dir
        if args.num_epochs:
            config_dict["num_epochs"] = args.num_epochs
        if args.batch_size:
            config_dict["batch_size"] = args.batch_size
        if args.learning_rate:
            config_dict["learning_rate"] = args.learning_rate
        if args.temperature:
            config_dict["temperature"] = args.temperature
        if args.alpha:
            config_dict["alpha"] = args.alpha
        if args.beta:
            config_dict["beta"] = args.beta
            
        config = DistillationConfig(**config_dict)
        trainer = DistillationTrainer(config)
    
    # Start training
    logger.info("Starting distillation training...")
    logger.info(f"Teacher model: {trainer.config.teacher_model}")
    logger.info(f"Student model: {trainer.config.student_model}")
    logger.info(f"Dataset: {trainer.config.dataset_name}")
    logger.info(f"Output directory: {trainer.config.output_dir}")
    
    trainer.train()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()