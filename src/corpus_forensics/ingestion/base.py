"""Base document loader interface and extension-based loader registry."""

from abc import ABC, abstractmethod
from pathlib import Path

from corpus_forensics.schemas import Document


class DocumentLoader(ABC):
    """Abstract base class for all document loaders."""

    @abstractmethod
    def load(self, path: Path | str) -> Document:
        """Load a file from the given path and return a Document instance.

        Args:
            path: Path to the target document.

        Returns:
            Document containing raw_content, visible_text, and hidden_content channel.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier name of the loader."""
        pass


_LOADER_REGISTRY: dict[str, type[DocumentLoader]] = {}


def register_loader(extensions: list[str], loader_cls: type[DocumentLoader]) -> None:
    """Register a DocumentLoader class for specific file extensions.

    Args:
        extensions: List of extensions including dot (e.g., ['.txt', '.md']).
        loader_cls: The loader class to instantiate.
    """
    for ext in extensions:
        _LOADER_REGISTRY[ext.lower()] = loader_cls


def get_loader_for_path(path: Path | str) -> DocumentLoader:
    """Resolve and return an instantiated DocumentLoader based on the file extension.

    Args:
        path: Path to the document file.

    Returns:
        Instantiated DocumentLoader.

    Raises:
        ValueError: If no loader is registered for the file extension.
    """
    file_path = Path(path)
    ext = file_path.suffix.lower()
    if ext not in _LOADER_REGISTRY:
        raise ValueError(
            f"Unsupported file extension '{ext}' for path '{path}'. "
            f"Registered extensions: {list(_LOADER_REGISTRY.keys())}"
        )
    loader_cls = _LOADER_REGISTRY[ext]
    return loader_cls()
