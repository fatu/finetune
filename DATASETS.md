# Valid Dataset Names Found in Codebase

Based on the search through the codebase, here are the valid dataset names that are actually used:

## Math Datasets
- `AI-MO/NuminaMath-TIR` - Used in grpo_math_trainer.py for math training
- `ai2-adapt-dev/gsm8k-train-round2` - Used in distillation example
- `ai2-adapt-dev/rlvr_gsm8k_zs` - Used in PPO training script
- `AI-MO/NuminaMath-CoT` - Referenced in dl_dataset.sh
- `meta-math/MetaMathQA` - Referenced in dl_dataset.sh
- `TIGER-Lab/MathInstruct` - Referenced in dl_dataset.sh
- `microsoft/orca-math-word-problems-200k` - Referenced in dl_dataset.sh
- `KbsdJames/Omni-MATH` - Referenced in dl_dataset.sh

## Conversation Datasets
- `neuralmagic/ultrachat_2k` - Used in grpo_conversation_simple.py
- `trl-lib/tldr` - Used in debug_trl_grpo.py and GRPO trainer examples
- `trl-lib/Capybara` - Referenced in dl_dataset.sh

## AppWorld Datasets
For AppWorld evaluation, the following dataset splits are available:
- `train` - Training split
- `dev` - Development split
- `test_normal` - Normal test split
- `test_challenge` - Challenging test split

## Usage Examples

### Math Training
```bash
python finetune/grpo_math_trainer.py \
    --model_name_or_path model/Qwen/Qwen3-4B \
    --dataset_name ai2-adapt-dev/gsm8k-train-round2 \
    --output_dir output/Qwen3-4B-GRPO-test
```

### Conversation Training
```bash
python finetune/grpo_conversation_simple.py
# This uses neuralmagic/ultrachat_2k by default
```

### Download Dataset
```bash
tempdata="neuralmagic/ultrachat_2k" bash scripts/dataset/dl_dataset.sh
```

## Note
These datasets need to be downloaded before use. Use the download script or ensure they're available in your HuggingFace cache.