from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

@dataclass
class ModelConfig:
    model_name_or_path: Optional[str] = None
    """The model checkpoint for weights initialization."""
    model_revision: Optional[str] = None
    """The specific model version to use (can be a branch name, tag name or commit id)."""

