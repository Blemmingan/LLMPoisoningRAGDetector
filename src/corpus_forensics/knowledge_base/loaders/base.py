"""Base interface for Knowledge Base attack source loaders."""

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from corpus_forensics.knowledge_base.schemas import AttackEntry

# Metadata flag for entries whose attack type has not been labeled yet.
LABEL_STATUS_KEY = "etiquetado"
LABEL_PENDING = "pendiente"


def normalize_text(text: str) -> str:
    """Normalization used only for de-duplication (whitespace and case)."""
    return " ".join(text.split()).lower()


def make_entry_id(source: str, split: str, text: str) -> str:
    """Stable entry ID derived from the source, split and normalized text."""
    digest = hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()[:12]
    return f"{source}-{split}-{digest}"


def scalar_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Keep only scalar, non-null values (the vector store rejects anything else)."""
    return {
        k: v for k, v in metadata.items() if isinstance(v, str | int | float | bool) and v != ""
    }


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
