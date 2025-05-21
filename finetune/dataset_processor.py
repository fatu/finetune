# this file deals with dataset pre-processing before training


@dataclass
class TokenizerConfig:

    tokenizer_name_or_path: Optional[str] = None
    tokenizer_revision: Optional[str] = None