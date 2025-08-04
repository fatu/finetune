# finetune

A comprehensive training framework for fine-tuning language models with various methods including PPO, GRPO, and Knowledge Distillation.

## Installation

### Install UV package manager
```bash
pip install uv --break-system-packages
```

### Install Flash Attention
```bash
uv pip install flash-attn --no-build-isolation
```

## Knowledge Distillation

Train a smaller student model using knowledge from a larger teacher model:

```bash
python finetune/distill/distillation_trainer.py \
      --teacher_model Qwen/Qwen3-1.7B \
      --student_model Qwen/Qwen3-0.6B \
      --dataset_name trl-lib/tldr \
      --output_dir output/distillation_qwen3_0.6b
```

### Configuration Options

- `--teacher_model`: Path or name of the teacher model
- `--student_model`: Path or name of the student model  
- `--dataset_name`: HuggingFace dataset to use for training
- `--output_dir`: Directory to save model checkpoints
- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Training batch size (default: 1)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--temperature`: Distillation temperature (default: 3.0)
- `--alpha`: Weight for distillation loss (default: 0.7)
- `--beta`: Weight for student loss (default: 0.3)
