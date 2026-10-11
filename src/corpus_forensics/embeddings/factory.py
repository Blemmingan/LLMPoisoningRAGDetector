"""Factory and registry for embedding model providers."""

from collections.abc import Callable
from typing import Any

from corpus_forensics.embeddings.base import Embedder
from corpus_forensics.embeddings.mock import MockEmbedder

_EMBEDDER_REGISTRY: dict[str, Callable[..., Embedder] | type[Embedder]] = {
    "mock": MockEmbedder,
}


def register_embedder(name: str, factory_or_cls: Callable[..., Embedder] | type[Embedder]) -> None:
    """Register a new embedder provider."""
    _EMBEDDER_REGISTRY[name.lower()] = factory_or_cls


def get_embedder(provider: str = "mock", **kwargs: Any) -> Embedder:
    """Instantiate and return an Embedder implementation by provider name.

    Args:
        provider: Provider name (e.g. 'mock', 'openai', 'huggingface').
        **kwargs: Provider-specific configuration arguments.

    Returns:
        Instantiated Embedder.

    Raises:
        ValueError: If provider is not registered.
    """
    key = provider.lower()
    if key not in _EMBEDDER_REGISTRY:
        raise ValueError(
            f"Unknown embedder provider '{provider}'. "
            f"Registered providers: {list(_EMBEDDER_REGISTRY.keys())}. "
            "To add a real embedder, register it with `register_embedder`."
        )
    factory_or_cls = _EMBEDDER_REGISTRY[key]
    return factory_or_cls(**kwargs)
