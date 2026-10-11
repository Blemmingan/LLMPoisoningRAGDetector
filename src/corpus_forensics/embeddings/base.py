"""Abstract base interface for text embedding providers."""

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Abstract base class for vector embedding models."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a batch of documents.

        Args:
            texts: List of document strings.

        Returns:
            List of float vector embeddings.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Compute embedding for a single query text.

        Args:
            text: Query string.

        Returns:
            Float vector embedding.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the vector space."""
        pass

    @property
    @abstractmethod
    def is_mock(self) -> bool:
        """Flag indicating if this is a mock implementation."""
        pass
