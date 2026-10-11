"""Abstract base interface for Vector Store backends."""

from abc import ABC, abstractmethod
from typing import Any

from corpus_forensics.schemas import RetrievedAttack


class VectorStore(ABC):
    """Abstract interface for storing and querying known attack vector embeddings."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add or upsert documents with explicit embeddings into the store.

        Args:
            ids: Unique entry identifiers.
            texts: Raw text contents of the attack entries.
            embeddings: Pre-computed vector embeddings.
            metadatas: Associated metadata dictionaries.
        """
        pass

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedAttack]:
        """Query nearest attack vectors given a query embedding.

        Args:
            query_embedding: Query vector.
            top_k: Number of nearest matches to retrieve.
            where: Optional metadata filter dict.

        Returns:
            List of RetrievedAttack items sorted by similarity.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the total number of entries in the store."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Delete all entries in the store or reset the collection."""
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics (e.g. entry counts by source, attack type)."""
        pass
