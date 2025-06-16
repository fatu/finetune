# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure
- `finetune/` - Main package with model utilities and dataset processing
- `scripts/` - Training scripts
- `local_dataset_cache/` - Cache for datasets

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.

## Common Commands

### Model and Dataset Download
```bash
# Download a model from HuggingFace (uses HF_ENDPOINT=https://hf-mirror.com)
tempmodel="model_name" bash scripts/download/dl_model.sh

# Download a dataset from HuggingFace
tempdata="dataset_name" bash scripts/dataset/dl_dataset.sh
```

### Training Commands
```bash
# Run PPO training (example with small model)
bash scripts/train/ppo_only.sh

# Run GRPO training for math tasks
python finetune/grpo_math_trainer.py \
    --model_name_or_path model/Qwen/Qwen3-0.6B \
    --dataset_name ai2-adapt-dev/gsm8k-train-round2 \
    --output_dir output/Qwen3-0.6B-GRPO-test \
    --num_train_epochs 1 \
    --logging_first_step true \
    --per_device_train_batch_size 2 \
    --gradient_checkpointing true \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 2 \
    --seed 42 \
    --num_grpo_epochs 2 \
    --logging_steps 1
```

### AppWorld Evaluation
```bash
# Run AppWorld evaluation
python finetune/appworld_eval.py

# Test AppWorld environment
python finetune/appworld_test.py
```

## High-Level Architecture

### Training Framework
The project is built on TRL (Transformer Reinforcement Learning) library and supports multiple training approaches:
- **PPO (Proximal Policy Optimization)**: Main RL training method in `finetune/ppo_only.py`
- **GRPO (Generalized Reward-based Policy Optimization)**: Math-focused training in `finetune/grpo_math_trainer.py`
- **Other methods**: DPO, KTO, CPO, ORPO available through TRL integration

### Key Components
1. **Model Loading**: Uses `finetune/model_utils.py` for loading models with PEFT/LoRA support
2. **Dataset Processing**: `finetune/dataset_processor.py` handles data loading and preprocessing
3. **Reward System**: Custom reward functions for math verification and AppWorld task completion
4. **VLLM Integration**: `finetune/vllm_utils.py` provides efficient inference during training
5. **AppWorld Environment**: `finetune/appworld_env.py` interfaces with the AppWorld benchmark for multi-step task evaluation

### Training Pipeline
1. Models are downloaded to `model/` directory
2. Datasets are cached in `local_dataset_cache/`
3. Training outputs go to `output/` with model checkpoints
4. TensorBoard logs are saved in `runs/`
5. Experiment results are stored in `experiments/outputs/`

### Configuration
- DeepSpeed configs in `finetune/ds_zero2_no_offload.json` and `finetune/trl_main/accelerate_configs/`
- Training uses mixed precision (bf16/fp16) by default
- Supports both single-GPU and multi-GPU training via Accelerate

### Dependencies
- Python >= 3.11.12
- PyTorch ecosystem (torch, transformers, accelerate, deepspeed)
- VLLM for efficient inference
- Ray for distributed computing
- Flash Attention 2 for performance (optional)