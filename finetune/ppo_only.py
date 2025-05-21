import time
import os

from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional, Tuple, Union

from rich.pretty import pprint

from finetune.utils import (
    ArgumentParserPlus
)

from finetune.model_utils import (
    ModelConfig
)

from finetune.dataset_transformation import (
    TokenizerConfig,
    get_cached_dataset_tulu
)

@dataclass
class Args:
    # required dataset args
    dataset_mixer_list: List[str] = None
    """A list of datasets (local or HF) to sample from."""
    dataset_mixer_eval_list: List[str] = None
    """A list of datasets (local or HF) to sample from for evaluation."""
    dataset_mixer_list_splits: List[str] = None
    """The dataset splits to use for training"""
    dataset_mixer_eval_list_splits: Optional[List[str]] = None
    """The dataset splits to use for evaluation"""
    dataset_transform_fn: list[str] = field(default_factory=lambda: ["rlvr_tokenize_v1", "rlvr_filter_v1"])
    """The list of transform functions to apply to the dataset."""
    dataset_cache_mode: Literal["hf", "local"] = "local"
    """The mode to use for caching the dataset."""
    dataset_local_cache_dir: str = "local_dataset_cache"
    """The directory to save the local dataset cache to."""
    dataset_config_hash: Optional[str] = None
    """The hash of the dataset configuration."""
    dataset_config_eval_hash: Optional[str] = None
    """The hash of the dataset configuration for evaluation."""
    dataset_skip_cache: bool = False
    """Whether to skip the cache."""
    max_token_length: int = 512
    """The maximum token length to use for the dataset"""
    max_prompt_token_length: int = 256
    """The maximum prompt token length to use for the dataset"""

    # various batch sizes
    gradient_accumulation_steps: Optional[int] = None
    """The number of gradient accumulation steps"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_rollout_batch_size: int = 64
    """The number of rollout episodes per iteration per device"""
    local_total_prompts: Optional[int] = None
    """The total number of prompts per device"""
    rollout_batch_size: Optional[int] = None
    """The number of rollout episodes per iteration"""
    local_dataloader_batch_size: Optional[int] = None
    """The batch size per GPU for the dataloader"""

    # online settings:
    num_epochs: int = 4
    """the number of epochs to train"""
    num_mini_batches: int = 1
    """Number of minibatches to split a batch into"""
    local_mini_batch_size: int = 64
    """the mini batch size per GPU"""

    reward_model_path: str = "EleutherAI/pythia-160m"
    """the path to the reward model"""
    reward_model_revision: Optional[str] = None
    """the revision of the reward model"""

    # online PPO specific args
    beta: float = 0.05
    """the beta value of the RLHF objective (KL coefficient)"""
    whiten_rewards: bool = False
    """whether to whiten the rewards"""

    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    seed: int = 1
    """Seed of the experiment"""
    run_name: Optional[str] = None
    """A unique name of this run"""
    output_dir: str = "output"
    """Where to save the model"""

def main(args: Args, tc: TokenizerConfig, model_config: ModelConfig):
    # ------------------------------------------------------------
    # Setup tokenizer
    tc.tokenizer_revision = model_config.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision
    tc.tokenizer_name_or_path = (
        model_config.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    if (
        tc.tokenizer_revision != model_config.model_revision
        and tc.tokenizer_name_or_path != model_config.model_name_or_path
    ):
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tc.tokenizer_revision=}` is different
                   from the model revision `{model_config.model_revision=}` or the tokenizer name `{tc.tokenizer_name_or_path=}`
                   is different from the model name `{model_config.model_name_or_path=}`."""
        print(warning)
    # some more runtime logging
    pprint([args, tc, model_config])
    tokenizer = tc.tokenizer
    # ------------------------------------------------------------
    # Set up runtime variables
    args.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    args.gradient_accumulation_steps = args.local_mini_batch_size
    args.micro_batch_size = args.local_mini_batch_size

    # PPO logic: do checks and set up dataloader batch size
    if args.whiten_rewards:
        assert (
            args.local_mini_batch_size >= 8
        ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
    args.local_dataloader_batch_size = args.rollout_batch_size

    # ------------------------------------------------------------
    # Set up datasets
    transform_fn_args = [
        {},
        {
            "max_token_length": args.max_token_length,
            "max_prompt_token_length": args.max_prompt_token_length,
        },
    ]
    train_dataset = get_cached_dataset_tulu(
        dataset_mixer_list=args.dataset_mixer_list,
        dataset_mixer_list_splits=args.dataset_mixer_list_splits,
        tc=tc,
        dataset_transform_fn=args.dataset_transform_fn,
        transform_fn_args=transform_fn_args,
        dataset_cache_mode=args.dataset_cache_mode,
        dataset_config_hash=args.dataset_config_hash,
        hf_entity=None,
        dataset_local_cache_dir=args.dataset_local_cache_dir,
        dataset_skip_cache=args.dataset_skip_cache,
    )


if __name__ == "__main__":
    parser = ArgumentParserPlus((Args, TokenizerConfig, ModelConfig))
    main(*parser.parse())