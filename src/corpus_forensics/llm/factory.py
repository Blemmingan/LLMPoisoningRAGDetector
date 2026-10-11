"""Factory and registry for LLM client providers."""

from collections.abc import Callable
from typing import Any

from corpus_forensics.llm.base import LLMClient
from corpus_forensics.llm.mock import MockLLMClient

_LLM_REGISTRY: dict[str, Callable[..., LLMClient] | type[LLMClient]] = {
    "mock": MockLLMClient,
}


def register_llm_client(
    name: str, factory_or_cls: Callable[..., LLMClient] | type[LLMClient]
) -> None:
    """Register a new LLM provider."""
    _LLM_REGISTRY[name.lower()] = factory_or_cls


def get_llm_client(provider: str = "mock", **kwargs: Any) -> LLMClient:
    """Instantiate and return an LLMClient implementation by provider name.

    Args:
        provider: Provider name (e.g. 'mock', 'openai', 'anthropic').
        **kwargs: Provider-specific configuration arguments.

    Returns:
        Instantiated LLMClient.

    Raises:
        ValueError: If provider is not registered.
    """
    key = provider.lower()
    if key not in _LLM_REGISTRY:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Registered providers: {list(_LLM_REGISTRY.keys())}. "
            "To add a real LLM, register it with `register_llm_client`."
        )
    factory_or_cls = _LLM_REGISTRY[key]
    return factory_or_cls(**kwargs)
