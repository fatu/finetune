import itertools
import logging
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import pandas as pd
import torch

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from finetune.ground_truth_utils import REWARD_FN_MAPPING

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_name_or_path: Optional[str] = None
    """The model checkpoint for weights initialization."""
    model_revision: Optional[str] = None
    """The specific model version to use (can be a branch name, tag name or commit id)."""
    torch_dtype: Optional[str] = None
    """Override the default `torch.dtype` and load the model under this dtype."""
    attn_implementation: Optional[Literal["flash_attention_2"]] = None
    """Which attention implementation to use; you can run --attn_implementation=flash_attention_2, in which case
    you must install this manually by running `pip install flash-attn --no-build-isolation`"""
    use_cache: Optional[bool] = None
    """Whether to use cache in the model."""
    gradient_checkpointing: bool = False
    """Whether to use gradient checkpointing in the model."""


# ----------------------------------------------------------------------------
# Model utilities; reward model stuff
def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

def first_true_indices(bools: torch.Tensor, dtype=torch.long) -> torch.Tensor:
    """
    Finds the index of the first `True` value in each row of a boolean tensor. If no `True` value exists in a row,
    it returns the length of the row.

    Args:
        bools (torch.Tensor): A boolean tensor of shape (batch_size, sequence_length), where `True` values indicate
                              the positions of interest.
        dtype (torch.dtype): The data type to use for the output indices (default is torch.long).

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the index of the first `True` value in each row.
                      If a row has no `True` value, the index will be the length of the row.
    """

    # Get the length of each row (i.e., the number of columns in the last dimension)
    # row_len is a scalar representing the length of each sequence (sequence_length)
    row_len = bools.size(-1)

    # Calculate the index positions for the first `True` in each row
    # ~bools: Invert the boolean values (True becomes False and vice versa)
    # ~bools.type(dtype): Convert the inverted boolean tensor to the specified dtype (0 for True, 1 for False)
    # row_len * (~bools).type(dtype): For `False` values, this will give `row_len`, for `True` values it gives 0.
    # torch.arange(row_len, dtype=dtype, device=bools.device): Generates a tensor with values [0, 1, 2, ..., row_len-1]
    # for each row. Shape: (sequence_length,)
    # zero_or_index: Shape (batch_size, sequence_length). This tensor contains the indices for `True` values and `row_len`
    # for `False` values.
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)

    # Return the minimum value in each row (i.e., the first `True` index or `row_len` if none exist)
    # torch.min(zero_or_index, dim=-1).values: This returns the minimum value in each row, which corresponds to the first
    # `True` value's index or `row_len` if there is no `True` in that row.
    # The returned tensor has shape (batch_size,)
    return torch.min(zero_or_index, dim=-1).values

def get_reward(
    model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int, context_length: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function computes reward scores for a batch of query responses based on a pre-trained reward model.

    Args:
        model (torch.nn.Module): The pre-trained reward model.
        query_responses (torch.Tensor): Tensor containing the tokenized responses for which to compute rewards.
            Shape: (batch_size, sequence_length)
        pad_token_id (int): The ID used for padding tokens in the tokenized sequences.
        context_length (int): The length of the prompt or context preceding the completions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - reward_logits: The logits output from the model for all tokens in the sequences.
              Shape: (batch_size, sequence_length)
            - final_scores: The final reward scores, one for each sequence, after adjusting for sequence lengths.
              Shape: (batch_size,)
            - sequence_lengths: The lengths of each sequence (excluding padding).
              Shape: (batch_size,)
    """

    # Create an attention mask where tokens that are not padding have a value of 1, and padding tokens have a value of 0
    # Shape: (batch_size, sequence_length)
    attention_mask = query_responses != pad_token_id

    # Calculate position IDs for each token, considering the cumulative sum of the attention mask (to exclude padding)
    # Shape: (batch_size, sequence_length)
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum

    # Access the LM backbone from the reward model using its base model prefix
    lm_backbone = getattr(model, model.base_model_prefix)

    # Replace padding tokens with zeros in the input IDs (so padding tokens won't affect the model's processing)
    # Shape: (batch_size, sequence_length)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,  # otherwise mistral-based RM would error out
    )
    reward_logits = model.score(output.hidden_states[-1])  # (batch_size, sequence_length)

    # Calculate the length of each sequence by finding the first occurrence of a padding token after the context
    # sequence_lengths shape: (batch_size,)
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
    assert (
        reward_logits.shape[-1] == 1
    ), "Reward model should output a single scalar per token. Check if you added `num_labels=1` when doing `AutoModelForSequenceClassification.from_pretrained(...)`."
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454

    # Return the reward logits for all tokens, the final reward scores for each sequence, and the sequence lengths
    return (
        # reward_logits shape: (batch_size, sequence_length)
        reward_logits,
        # final_scores shape: (batch_size,)
        reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
        ].squeeze(
            -1
        ),  # Shape: (batch_size,)
        sequence_lengths,
    )

def apply_verifiable_reward(
    responses: List[torch.Tensor],
    decoded_responses: List[str],
    ground_truths: List[str],
    datasets: List[Union[str, List[str]]],
    reward_mult: int = 10,
):
    rewards = []
    per_func_rewards = []
    for tok_prediction, prediction, ground_truth, dataset in zip(
        responses, decoded_responses, ground_truths, datasets
    ):
        # allow multiple ground truths and datasets for a single response
        if isinstance(ground_truth, str):
            ground_truth_list = [ground_truth]
        else:
            ground_truth_list = ground_truth
        if isinstance(dataset, str):
            dataset_list = [dataset]
        else:
            dataset_list = dataset
        assert len(ground_truth_list) == len(dataset_list), "Ground truth and dataset list lengths do not match."
        # for now, we just assume rewards are additive, rather than more complex functions.
        reward = 0
        per_func_reward = {}
        for gt, ds in zip(ground_truth_list, dataset_list):
            reward_func = REWARD_FN_MAPPING.get(ds.lower())
            if reward_func is None:
                logger.warning("No reward function found for dataset %s. Skipping reward.", ds)
                continue
            reward_weight = reward_func.weight
            # compare with ground truth.
            # sometimes we need the tokenized pred.
            reward_result = reward_func(
                tokenized_prediction=tok_prediction,
                prediction=prediction,
                label=gt,
            )
            logger.info("Applying ground truth reward 🤗")
            reward += reward_mult * reward_result * reward_weight
            per_func_reward[ds] = per_func_reward.get(ds, 0) + (reward_mult * reward_result * reward_weight)
        rewards.append(reward)
        per_func_rewards.append(per_func_reward)
    return rewards, per_func_rewards

def exact_div(a, b, custom_error_message=""):
    q = a // b
    if a != q * b:
        raise ValueError(f"{custom_error_message}, inexact division: {a} / {b} = {a / b}")
    return q

@torch.compile(dynamic=True)
def log_softmax_and_gather(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    torch compiled version of the common `log_softmax -> gather` operation.

    The compiled version of this opration avoids the (significant) memory overhead of
    allocating a new (batch_size, seq_len, vocab_size) tensor to store the logprobs.

    See https://github.com/allenai/open-instruct/pull/584
    """
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


def truncate_response(stop_token_id: int, pad_token_id: int, responses: torch.Tensor):
    """
    Truncates the responses at the first occurrence of the stop token, filling the rest with pad tokens.
    Args:
        stop_token_id (`int`):
            The token ID representing the stop token where truncation occurs.
        pad_token_id (`int`):
            The token ID representing the pad token used to fill the truncated responses.
        responses (`torch.Tensor`):
            The tensor containing the responses to be truncated.
    Returns:
        `torch.Tensor`:
            The truncated responses tensor with pad tokens filled after the stop token.
    """
    trunc_idxs = first_true_indices(responses == stop_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
    idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, pad_token_id)
    return postprocessed_responses


# ----------------------------------------------------------------------------
# Quality of life utilities
def print_rich_table(df: pd.DataFrame) -> Table:
    console = Console()
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.print(table)

def print_rich_single_line_metrics(metrics):
    # Create main table
    table = Table(show_header=False, box=None)
    table.add_column("Category", style="cyan")
    table.add_column("Values", style="magenta")

    # Group metrics by their prefix
    grouped_metrics = defaultdict(list)
    for key, value in metrics.items():
        category = key.split("/")[0] if "/" in key else "other"
        grouped_metrics[category].append((key, value))

    # Sort groups by category name
    for category in sorted(grouped_metrics.keys()):
        values = grouped_metrics[category]
        value_strings = []
        for key, value in values:
            # Use the last part of the key as the display name
            display_name = key.split("/")[-1]
            value_strings.append(f"{display_name}: {format_value(value)}")

        # Join all values for this category into a single string
        values_str = " | ".join(value_strings)
        table.add_row(category, values_str)

    # Create a panel with the table
    panel = Panel(
        table,
        title="Metrics",
        expand=False,
        border_style="bold green",
    )

    # Print the panel
    rprint(panel)