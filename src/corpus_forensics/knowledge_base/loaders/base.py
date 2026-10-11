"""Base interface for Knowledge Base attack source loaders."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from corpus_forensics.knowledge_base.schemas import AttackEntry


class BaseKBLoader(ABC):
    """Abstract loader for ingesting attack corpus entries from external datasets."""

    @abstractmethod
    def load(self, source_path: Path | str | None = None) -> Iterable[AttackEntry]:
        """Load attack entries from source data.

        Args:
            source_path: Optional path to data file or directory.

        Returns:
            Iterable of AttackEntry models.
        """
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Source identifier name."""
        pass
