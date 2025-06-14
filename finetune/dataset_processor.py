# this file deals with dataset pre-processing before training


from dataclasses import dataclass
from typing import Optional, Union

import torch


INPUT_IDS_PROMPT_KEY = "input_ids_prompt"
GROUND_TRUTHS_KEY = "ground_truth"
DATASET_SOURCE_KEY = "dataset"

@dataclass
class TokenizerConfig:

    tokenizer_name_or_path: Optional[str] = None
    tokenizer_revision: Optional[str] = None


class SimpleGenerateCollatorWithGroundTruth:
    """Simple collator for generation task (always pad from the LEFT)"""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]):
        """the input will have input_ids_prompt"""
        # Find max length in the batch
        max_length = -1
        for i in range(len(batch)):
            max_length = max(max_length, len(batch[i][INPUT_IDS_PROMPT_KEY]))
        assert max_length > 0, "the dataset is empty"

        # Initialize lists to store padded sequences and attention masks
        padded_sequences = []

        for i in range(len(batch)):
            # Calculate padding length
            pad_length = max_length - len(batch[i][INPUT_IDS_PROMPT_KEY])

            # Pad from the left
            padding = [self.pad_token_id] * pad_length
            padded_sequence = padding + batch[i][INPUT_IDS_PROMPT_KEY]
            padded_sequences.append(padded_sequence)

        # Convert to tensors
        padded_sequences = torch.tensor(padded_sequences)

        # ground truths
        ground_truths = [x[GROUND_TRUTHS_KEY] for x in batch]

        # datasets
        datasets = [x[DATASET_SOURCE_KEY] for x in batch]

        return {
            INPUT_IDS_PROMPT_KEY: padded_sequences,
            GROUND_TRUTHS_KEY: ground_truths,
            DATASET_SOURCE_KEY: datasets,
        }