#!/bin/bash

# Example script for running distillation training

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com

# Run distillation training with config file
python finetune/distill/distillation_trainer.py \
    --config finetune/distill/config/distillation_config.json \
    --output_dir output/distillation_qwen3_0.6b \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-5

# Alternative: Run with command-line arguments
# python finetune/distill/distillation_trainer.py \
#     --teacher_model model/Qwen/Qwen3-1.7B \
#     --student_model model/Qwen/Qwen3-0.6B \
#     --dataset_name trl-lib/tldr \
#     --output_dir output/distillation_qwen3_0.6b \
#     --num_epochs 3 \
#     --batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 5e-5 \
#     --temperature 3.0 \
#     --alpha 0.7 \
#     --beta 0.3

# Multi-GPU training with accelerate
# accelerate launch --config_file finetune/trl_main/accelerate_configs/multi_gpu.yaml \
#     finetune/distill/distillation_trainer.py \
#     --config finetune/distill/config/distillation_config.json