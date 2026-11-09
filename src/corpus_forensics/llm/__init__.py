"""LLM client module."""

from corpus_forensics.llm.base import (
    BudgetExceededError,
    LLMClient,
    LLMError,
    ProviderQuotaError,
)
from corpus_forensics.llm.factory import get_llm_client, register_llm_client
from corpus_forensics.llm.litellm_client import LiteLLMClient
from corpus_forensics.llm.mock import MockLLMClient

__all__ = [
    "BudgetExceededError",
    "LLMClient",
    "LLMError",
    "ProviderQuotaError",
    "LiteLLMClient",
    "MockLLMClient",
    "get_llm_client",
    "register_llm_client",
]
