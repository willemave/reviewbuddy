"""Token estimation helpers for synthesis budgeting."""

from __future__ import annotations

import math
from functools import lru_cache

import tiktoken

DEFAULT_ENCODING_NAME = "o200k_base"
ESTIMATE_MARGIN = 0.10
ESTIMATE_FIXED_OVERHEAD = 32


@lru_cache(maxsize=16)
def _encoding_name_for_model(model_name: str) -> str:
    try:
        return tiktoken.encoding_for_model(model_name).name
    except KeyError:
        return DEFAULT_ENCODING_NAME


@lru_cache(maxsize=16)
def _encoding_for_model(model_name: str):
    return tiktoken.get_encoding(_encoding_name_for_model(model_name))


def count_tokens(text: str, model_name: str = "gpt-5.4") -> int:
    """Count tokens using a stable tokenizer mapping."""

    if not text:
        return 0
    return len(_encoding_for_model(model_name).encode(text))


def estimate_tokens(text: str, model_name: str = "gpt-5.4") -> int:
    """Estimate prompt tokens conservatively for budgeting."""

    raw_count = count_tokens(text, model_name=model_name)
    if raw_count == 0:
        return 0
    return raw_count + math.ceil(raw_count * ESTIMATE_MARGIN) + ESTIMATE_FIXED_OVERHEAD
