"""Token counting for the analyst prompt (estimates before any paid call).

Counts use tiktoken: `o200k_base` for the GPT-4o / 4.1 family (the course model is
gpt-4o-mini) and `cl100k_base` otherwise. The structured-output JSON schema sent with
each call is counted too, since it is part of the input.
"""

import json
from functools import lru_cache
from typing import Any

from corpus_forensics.schemas import AnalisisChunkLLM

O200K_PREFIXES = ("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4")


@lru_cache(maxsize=4)
def _encoding(name: str) -> Any:
    import tiktoken  # noqa: PLC0415

    return tiktoken.get_encoding(name)


def encoding_for(model: str | None) -> str:
    base = (model or "gpt-4o-mini").split("/")[-1]
    return "o200k_base" if base.startswith(O200K_PREFIXES) else "cl100k_base"


def count_tokens(text: str, model: str | None = None) -> int:
    """Tokens of a text for the given model (special-token strings counted as text)."""
    return len(_encoding(encoding_for(model)).encode(text, disallowed_special=()))


@lru_cache(maxsize=4)
def schema_tokens(model: str | None = None) -> int:
    """Tokens of the analyst's structured-output schema (sent with every call)."""
    schema = json.dumps(AnalisisChunkLLM.model_json_schema(), separators=(",", ":"))
    return count_tokens(schema, model)
