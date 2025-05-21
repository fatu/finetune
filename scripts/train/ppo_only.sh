python finetune/ppo_only.py \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 512 \
    --max_prompt_token_length 512 \
    --model_name_or_path EleutherAI/pythia-14m \
    --seed 3 \
    --local_mini_batch_size 4

    