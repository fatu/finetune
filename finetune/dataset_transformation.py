
# this file deals with dataset pre-processing before training

import copy
import hashlib
import json
import multiprocessing
import os
import re
from concurrent import futures
from pathlib import Path
from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import List, Literal, Optional, Any, Dict

import torch
import transformers
from datasets import Dataset, concatenate_datasets, load_dataset

from rich.console import Console
from rich.text import Text
from transformers import (
    AutoTokenizer,
    AutoConfig,
    GPTNeoXTokenizerFast,
    PreTrainedTokenizer,
)
from transformers.utils.hub import (
    _CACHED_NO_EXIST,
    TRANSFORMERS_CACHE,
    extract_commit_hash,
    try_to_load_from_cache,
)

from huggingface_hub.file_download import REGEX_COMMIT_HASH

COLORS = ["on red", "on green", "on blue", "on yellow", "on magenta"]

def visualize_token(tokens: list[int], tokenizer: PreTrainedTokenizer):
    i = 0
    console = Console()
    rich_text = Text()
    for i, token in enumerate(tokens):
        color = COLORS[i % len(COLORS)]
        decoded_token = tokenizer.decode(token)
        rich_text.append(f"{decoded_token}", style=color)
    console.print(rich_text)

# ----------------------------------------------------------------------------
# Utilities
def custom_cached_file(model_name_or_path: str, filename: str, revision: str = None, repo_type: str = "model"):
    """@vwxyzjn: HF's `cached_file` no longer works for `repo_type="dataset"`."""
    # local_file = os.path.join(model_name_or_path, filename)

    if os.path.isdir(model_name_or_path):
        resolved_file = os.path.join(model_name_or_path, filename)
        if os.path.isfile(resolved_file):
            return resolved_file
        else:
            return None
    else:
        resolved_file = try_to_load_from_cache(
            model_name_or_path, filename, cache_dir=TRANSFORMERS_CACHE, revision=revision, repo_type=repo_type
        )
        # special return value from try_to_load_from_cache
        if resolved_file == _CACHED_NO_EXIST:
            return None
        return resolved_file

def get_commit_hash(
    model_name_or_path: str, revision: str, filename: str = "config.json", repo_type: str = "model"
) -> str:
    file = custom_cached_file(model_name_or_path, filename, revision=revision, repo_type=repo_type)
    commit_hash = extract_commit_hash(file, None)
    return commit_hash

# Performance tuning. Some rough numbers:
APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU = 400
FILTER_EXAMPLE_PER_SECOND_PER_CPU = 1130

def get_num_proc(dataset_len: int, num_available_cpus: int, example_per_second_per_cpu) -> int:
    num_required_cpus = max(1, dataset_len // example_per_second_per_cpu)
    return min(num_required_cpus, num_available_cpus)

# ----------------------------------------------------------------------------
# Tokenization
# Chat templates
# flake8: noqa
# note we added `{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}`
# because we want the template to not output eos_token if `add_generation_prompt=True`
CHAT_TEMPLATES = {
    "simple_concat_with_space": (
        "{% for message in messages %}"
        "{{ ' ' if not loop.first else '' }}"
        "{{ message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "simple_concat_with_new_line": (
        "{% for message in messages %}"
        "{{ '\n' if not loop.first else '' }}"
        "{{ message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "simple_chat": (
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': ' + message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "assistant_message_only": (
        "{% for message in messages %}"
        "{% if message['role'] == 'assistant' %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "zephyr": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + eos_token + '\n' }}"
        "{% elif message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + eos_token + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "tulu": (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{% if not loop.last %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
        "{% else %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token }}"
        "{% endif %}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "tulu_thinker": (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{% set content = message['content'] %}"
        "{% if not loop.last %}"
        "{{ '<|assistant|>\n' + content + eos_token + '\n' }}"
        "{% else %}"
        "{{ '<|assistant|>\n' + content + eos_token }}"
        "{% endif %}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n<think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "tulu_thinker_r1_style": (
        "A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in "
        "the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> "
        "and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> "
        "<answer> answer here </answer>."
        "\n\n"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{% set content = message['content'] %}"
        "{% if '</think>' in content %}"
        "{% set content = content.split('</think>')[-1] %}"
        "{% endif %}"
        "{% if not loop.last %}"
        "{{ '<|assistant|>\n' + content + eos_token + '\n' }}"
        "{% else %}"
        "{{ '<|assistant|>\n' + content + eos_token }}"
        "{% endif %}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n<think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    # template is taken from https://arxiv.org/abs/2501.12948.
    "r1_simple_chat": (
        "A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in "
        "the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> "
        "and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> "
        "<answer> answer here </answer>."
        "\n\n"
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': ' + message['content'] + '\n' }}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ 'Assistant:' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "r1_simple_chat_postpend_think": (
        "A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in "
        "the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> "
        "and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> "
        "<answer> answer here </answer>."
        "\n\n"
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': ' + message['content'] + '\n' }}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ 'Assistant: <think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "r1_simple_chat_postpend_think_orz_style": (
        "A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in "
        "the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> "
        "and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> "
        "<answer> answer here </answer>."
        "\n\n"
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\\\boxed{} tag. This is the problem: ' + message['content'] + '\n' }}"  # \\\\boxed{} is for jinja template escape
        "{% if loop.last and add_generation_prompt %}"
        "{{ 'Assistant: <think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "r1_simple_chat_postpend_think_tool_vllm": (
        "A conversation between User and Assistant. "
        "The User asks a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in "
        "the mind and then provides the User with the answer. "
        "\n\n"
        "When given a question, the Assistant must conduct reasoning inside the <think> "
        "and </think> tags. During reasoning, the Assistant may write and execute python "
        "code using the <code> </code> tag, in order to solve the problem or verify the answer. "
        "Then the Assistant will get the stdout and stderr in the <output> and </output> tags. "
        "For example, the code could be\n"
        "<code>\n"
        "x, y = 1, 2\n"
        "result = x + y\n"
        "print(result)\n"
        "</code>\n"
        "or\n"
        "<code>\n"
        "import sympy as sp\n"
        "from sympy import Symbol\n"
        "x = Symbol('x')\n"
        "y = Symbol('y')\n"
        "solution = sp.solve(x**2 + y**2 - 1, (x, y))\n"
        "print(solution)\n"
        "</code>\n"
        "The Assistant will always `print` the result of the code execution in order to see it in the <output> tag. "
        "The Assistant may use the <code> </code> tag multiple times. "
        "When the Assistant is done reasoning, it should provide the answer inside the <answer> "
        "and </answer> tag."
        "\n\n"
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\\\boxed{} tag. This is the problem: ' + message['content'] + '\n' }}"  # \\\\boxed{} is for jinjia template escape
        "{% if loop.last and add_generation_prompt %}"
        "{{ 'Assistant: <think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
}

def get_tokenizer_tulu_v2_2(tc: "TokenizerConfig"):
    config = AutoConfig.from_pretrained(tc.tokenizer_name_or_path, revision=tc.tokenizer_revision)

    tokenizer = AutoTokenizer.from_pretrained(
        tc.tokenizer_name_or_path,
        revision=tc.tokenizer_revision,
        trust_remote_code=tc.trust_remote_code,
        use_fast=tc.use_fast,
    )
    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    # only add if the pad token is not present already, or if the current one is set to eos_token_id.
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        if isinstance(tokenizer, GPTNeoXTokenizerFast):
            # OLMo newer models use this tokenizer
            if tokenizer.bos_token is None:
                tokenizer.bos_token = tokenizer.eos_token
                assert (
                    tc.add_bos
                ), "For OLMo with GPTNeoX, you must add bos token to the beginning of the input sequence."
            # else, pythia / other models
            else:
                num_added_tokens = tokenizer.add_special_tokens(
                    {
                        "pad_token": "<pad>",
                    }
                )
                assert (
                    num_added_tokens <= 1
                ), "GPTNeoXTokenizer should only add one special token - the pad_token (or no tokens if already set in SFT)."
    assert (
        tokenizer.pad_token_id != tokenizer.eos_token_id
    ), "pad token and eos token matching causes issues in our setup."

    # set the tokenizer chat template to the training format
    # this will be used for encoding the training examples
    # and saved together with the tokenizer to be used later.
    if tc.chat_template_name in CHAT_TEMPLATES:
        tokenizer.chat_template = CHAT_TEMPLATES[tc.chat_template_name]
    else:
        try:
            tokenizer.chat_template = AutoTokenizer.from_pretrained(tc.tokenizer_name_or_path).chat_template
        except Exception:
            raise ValueError(f"Could not find chat template for {tc.tokenizer_name_or_path}.")

    return tokenizer

GET_TOKENIZER_FN = {
    "get_tokenizer_tulu_v2_2": get_tokenizer_tulu_v2_2,
}

DEFAULT_SFT_MESSAGES_KEY = "messages"
GROUND_TRUTHS_KEY = "ground_truth"
DATASET_SOURCE_KEY = "dataset"

@dataclass
class TokenizerConfig:

    tokenizer_name_or_path: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    trust_remote_code: bool = False
    use_fast: bool = True
    chat_template_name: str = "tulu"
    add_bos: bool = False
    get_tokenizer_fn: str = "get_tokenizer_tulu_v2_2"


    # backward compatibility to make sure script runs
    use_slow_tokenizer: bool = False  # completely ignored
    tokenizer_name: Optional[str] = None
    ground_truths_key: str = GROUND_TRUTHS_KEY
    """columns name for the ground truth"""
    sft_messages_key: str = DEFAULT_SFT_MESSAGES_KEY
    """columns name for the sft messages"""

    @cached_property
    def tokenizer(self):
        return GET_TOKENIZER_FN[self.get_tokenizer_fn](self)

# ----------------------------------------------------------------------------
# Dataset Transformation
# SFT dataset
INPUT_IDS_KEY = "input_ids"
ATTENTION_MASK_KEY = "attention_mask"
LABELS_KEY = "labels"
TOKENIZED_SFT_DATASET_KEYS = [
    INPUT_IDS_KEY,
    ATTENTION_MASK_KEY,
    LABELS_KEY,
]

# Preference dataset
# NOTE (Costa): the `INPUT_IDS_PROMPT_KEY` is just for visualization purposes only
# also we don't really need `CHOSEN_ATTENTION_MASK_KEY` and `REJECTED_ATTENTION_MASK_KEY`
# since we are always padding from the right with a collator; however they might become
# more useful if we want to do some sort of packing in the future. The nice thing is
# that the tokenization logic would work for both DPO and RM training.
DEFAULT_CHOSEN_KEY = "chosen"
DEFAULT_REJECTED_KEY = "rejected"
CHOSEN_INPUT_IDS_KEY = "chosen_input_ids"
CHOSEN_ATTENTION_MASK_KEY = "chosen_attention_mask"
CHOSEN_LABELS_KEY = "chosen_labels"
REJECTED_INPUT_IDS_KEY = "rejected_input_ids"
REJECTED_ATTENTION_MASK_KEY = "rejected_attention_mask"
REJECTED_LABELS_KEY = "rejected_labels"

INPUT_IDS_PROMPT_KEY = "input_ids_prompt"
ATTENTION_MASK_PROMPT_KEY = "attention_mask_prompt"

TOKENIZED_PREFERENCE_DATASET_KEYS = [
    CHOSEN_INPUT_IDS_KEY,
    CHOSEN_LABELS_KEY,
    CHOSEN_ATTENTION_MASK_KEY,
    REJECTED_INPUT_IDS_KEY,
    REJECTED_LABELS_KEY,
    REJECTED_ATTENTION_MASK_KEY,
]

def rlvr_tokenize_v1(
    row: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    sft_messages_key: str = DEFAULT_SFT_MESSAGES_KEY,
    ground_truths_key: str = GROUND_TRUTHS_KEY,
    dataset_source_key: str = DATASET_SOURCE_KEY,
):
    if len(row[sft_messages_key]) == 1:
        prompt = row[sft_messages_key]
    else:
        prompt = row[sft_messages_key][:-1]
    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
    )
    row[INPUT_IDS_KEY] = tokenizer.apply_chat_template(row[sft_messages_key])
    row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])
    labels = copy.deepcopy(row[INPUT_IDS_KEY])
    row[LABELS_KEY] = labels
    row[GROUND_TRUTHS_KEY] = row[ground_truths_key]
    row[DATASET_SOURCE_KEY] = row[dataset_source_key]
    return row

def rlvr_filter_v1(
    row: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    need_contain_labels: bool = True,
    max_prompt_token_length: Optional[int] = None,
    max_token_length: Optional[int] = None,
):
    max_prompt_token_length_ok = True
    if max_prompt_token_length is not None:
        max_prompt_token_length_ok = len(row[INPUT_IDS_PROMPT_KEY]) <= max_prompt_token_length

    max_token_length_ok = True
    if max_token_length is not None:
        max_token_length_ok = len(row[INPUT_IDS_KEY]) <= max_token_length

    contain_some_labels = any(x != -100 for x in row[LABELS_KEY])
    return max_prompt_token_length_ok and max_token_length_ok and (contain_some_labels or not need_contain_labels)


TRANSFORM_FNS = {
    "rlvr_tokenize_v1": (rlvr_tokenize_v1, "map"),
    "rlvr_filter_v1": (rlvr_filter_v1, "filter"),
}
    
# ----------------------------------------------------------------------------
# Dataset Configuration and Caching
@dataclass
class DatasetConfig:
    dataset_name: str
    dataset_split: str
    dataset_revision: str
    dataset_range: Optional[int] = None
    transform_fn: List[str] = field(default_factory=list)
    transform_fn_args: List[Dict[str, Any]] = field(default_factory=list)
    target_columns: Optional[List[str]] = None

    # for tracking purposes
    dataset_commit_hash: Optional[str] = None

    def __post_init__(self):
        # if the file exists locally, use the local file
        if os.path.exists(self.dataset_name) and self.dataset_name.endswith(".jsonl"):
            assert self.dataset_split == "train", "Only train split is supported for local jsonl files."
            self.dataset = load_dataset(
                "json",
                data_files=self.dataset_name,
                split=self.dataset_split,
            )
        else:
            # commit hash only works for hf datasets
            self.dataset_commit_hash = get_commit_hash(
                self.dataset_name, self.dataset_revision, "README.md", "dataset"
            )
            self.dataset = load_dataset(
                self.dataset_name,
                split=self.dataset_split,
                revision=self.dataset_revision,
            )
        if self.dataset_range is None:
            dataset_range = len(self.dataset)
            self.update_range(dataset_range)

    def update_range(self, dataset_range: int):
        self.dataset_range = dataset_range
        if self.dataset_range > len(self.dataset):
            raise ValueError("Dataset range exceeds dataset length")
        self.dataset = self.dataset.select(range(self.dataset_range))

def get_dataset_v1(dc: DatasetConfig, tc: TokenizerConfig):
    assert len(dc.transform_fn) == len(
        dc.transform_fn_args
    ), f"transform_fn and transform_fn_args must have the same length: {dc.transform_fn=} != {dc.transform_fn_args=}"
    # beaker specific logic; we may get assigned 15.5 CPU, so we convert it to float then int
    num_proc = int(float(os.environ.get("BEAKER_ASSIGNED_CPU_COUNT", multiprocessing.cpu_count())))

    tokenizer = tc.tokenizer
    dataset = dc.dataset
    for fn_name, fn_args in zip(dc.transform_fn, dc.transform_fn_args):
        fn, fn_type = TRANSFORM_FNS[fn_name]
        # always pass in tokenizer and other args if needed
        fn_kwargs = {"tokenizer": tokenizer}
        fn_kwargs.update(fn_args)

        # perform the transformation
        target_columns = dataset.column_names if dc.target_columns is None else dc.target_columns
        if fn_type == "map":
            dataset = dataset.map(
                fn,
                fn_kwargs=fn_kwargs,
                remove_columns=[col for col in dataset.column_names if col not in target_columns],
                num_proc=get_num_proc(len(dataset), num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
            )
        elif fn_type == "filter":
            dataset = dataset.filter(
                fn,
                fn_kwargs=fn_kwargs,
                num_proc=get_num_proc(len(dataset), num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
            )
        # NOTE: elif we can implement packing here to create a packed SFT dataset. Low priority for now.
        else:
            raise ValueError(f"Unknown transform function type: {fn_type}")

    if len(dataset) == 0:
        raise ValueError("No examples left after transformation")
    return dataset

def compute_config_hash(dcs: List[DatasetConfig], tc: TokenizerConfig) -> str:
    """Compute a deterministic hash of both configs for caching."""
    dc_dicts = [{k: v for k, v in asdict(dc).items() if v is not None} for dc in dcs]
    tc_dict = {k: v for k, v in asdict(tc).items() if v is not None}
    combined_dict = {"dataset_configs": dc_dicts, "tokenizer_config": tc_dict}
    config_str = json.dumps(combined_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:10]

class LocalDatasetTransformationCache:
    def __init__(self, config_hash: str, dataset_local_cache_dir: str):
        """Initialize the local cache with a directory path."""
        self.config_hash = config_hash
        self.dataset_local_cache_dir = dataset_local_cache_dir
        os.makedirs(dataset_local_cache_dir, exist_ok=True)

    def get_cache_path(self) -> str:
        """Get the path to the cached dataset."""
        return os.path.join(self.dataset_local_cache_dir, self.config_hash)

    def save_config(self, config_hash: str, dcs: List[DatasetConfig], tc: TokenizerConfig):
        """Save the configuration to a JSON file."""
        config_path = os.path.join(self.get_cache_path(), "config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        config_dict = {
            "tokenizer_config": asdict(tc),
            "dataset_configs": [asdict(dc) for dc in dcs],
            "config_hash": config_hash,
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def load_or_transform_dataset(
        self, dcs: List[DatasetConfig], tc: TokenizerConfig, dataset_skip_cache: bool = False
    ) -> Dataset:
        """Load dataset from local cache if it exists, otherwise transform and cache it locally."""
        cache_path = self.get_cache_path()

        # Check if the cache exists
        if os.path.exists(cache_path) and not dataset_skip_cache:
            print(f"✅ Found cached dataset at {cache_path}")
            return Dataset.load_from_disk(cache_path, keep_in_memory=True)

        print(f"Cache not found or invalid, transforming datasets...")

        # Transform each dataset
        transformed_datasets = []
        for dc in dcs:
            dataset = get_dataset_v1(dc, tc)
            transformed_datasets.append(dataset)

        # Combine datasets
        combined_dataset = concatenate_datasets(transformed_datasets)
        if dataset_skip_cache:
            return combined_dataset

        # Save to local cache
        combined_dataset.save_to_disk(cache_path)
        self.save_config(self.config_hash, dcs, tc)
        print(f"🚀 Saved transformed dataset to {cache_path}")
        print(f"✅ Found cached dataset at {cache_path}")
        return Dataset.load_from_disk(cache_path, keep_in_memory=True)

def get_cached_dataset_tulu(
    dataset_mixer_list: List[str],
    dataset_mixer_list_splits: List[str],
    tc: TokenizerConfig,
    dataset_transform_fn: List[str],
    transform_fn_args: List[Dict[str, Any]],
    target_columns: Optional[List[str]] = None,
    dataset_cache_mode: Literal["hf", "local"] = "local",
    dataset_config_hash: Optional[str] = None,
    hf_entity=None,
    dataset_local_cache_dir: str = "local_dataset_cache",
    dataset_skip_cache: bool = False,
) -> Dataset:
    dcs = []
    if dataset_config_hash is None:
        if len(dataset_mixer_list_splits) == 1:
            print("by default, we will use the same split for all datasets")
            dataset_mixer_list_splits = [dataset_mixer_list_splits[0]] * len(dataset_mixer_list)
        else:
            if len(dataset_mixer_list_splits) != len(dataset_mixer_list):
                raise ValueError(
                    f"dataset_mixer_list_splits length must be the same as dataset_mixer_list: {len(dataset_mixer_list_splits)=} != {len(dataset_mixer_list)=}"
                )
        assert len(dataset_mixer_list) % 2 == 0, f"Data mixer list length is not even: {dataset_mixer_list}"
        for i in range(0, len(dataset_mixer_list), 2):
            dataset_name = dataset_mixer_list[i]
            frac_or_num_samples = dataset_mixer_list[i + 1]
            if "." in frac_or_num_samples:
                frac_or_num_samples = float(frac_or_num_samples)
            else:
                frac_or_num_samples = int(frac_or_num_samples)

            dataset_config = DatasetConfig(
                dataset_name=dataset_name,
                dataset_split=dataset_mixer_list_splits[i],
                dataset_revision="main",
                transform_fn=dataset_transform_fn,
                transform_fn_args=transform_fn_args,
                target_columns=target_columns,
            )
            if frac_or_num_samples > 1.0:
                new_range = int(frac_or_num_samples)
            else:
                new_range = int(frac_or_num_samples * len(dataset_config.dataset))
            dataset_config.update_range(new_range)
            dcs.append(dataset_config)
        dataset_config_hash = compute_config_hash(dcs, tc)
    if dataset_cache_mode == "local":
        cache = LocalDatasetTransformationCache(
            config_hash=dataset_config_hash, dataset_local_cache_dir=dataset_local_cache_dir
        )
    return cache.load_or_transform_dataset(dcs, tc, dataset_skip_cache=dataset_skip_cache)