"""Plain text and Markdown document loader (fully functional)."""

from pathlib import Path

from corpus_forensics.ingestion.base import DocumentLoader, register_loader
from corpus_forensics.schemas import Document


class TextLoader(DocumentLoader):
    """Loads plain text and markdown documents."""

    @property
    def name(self) -> str:
        return "text_loader"

    def load(self, path: Path | str) -> Document:
        """Load text or markdown document into a Document model.

        Args:
            path: Path to the plain text or markdown file.

        Returns:
            Document instance with visible_text and empty hidden_content.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        raw_content = file_path.read_text(encoding="utf-8", errors="replace")
        source_type = "markdown" if file_path.suffix.lower() in [".md", ".markdown"] else "text"

        return Document(
            id=str(file_path),
            raw_content=raw_content,
            visible_text=raw_content,
            hidden_content="",
            source_type=source_type,
            metadata={
                "file_name": file_path.name,
                "file_size_bytes": file_path.stat().st_size,
                "extension": file_path.suffix.lower(),
            },
        )


# Register text loader for common plain text and markdown extensions
register_loader([".txt", ".md", ".markdown"], TextLoader)
