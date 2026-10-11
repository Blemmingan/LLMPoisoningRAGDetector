"""Tests for text and markdown ingestion loader."""

from pathlib import Path

import pytest

from corpus_forensics.ingestion.base import get_loader_for_path
from corpus_forensics.ingestion.text_loader import TextLoader


def test_text_loader_functional(tmp_path: Path) -> None:
    """Verify loading text document."""
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Plain text content here.", encoding="utf-8")

    loader = TextLoader()
    doc = loader.load(txt_file)

    assert doc.id == str(txt_file)
    assert doc.visible_text == "Plain text content here."
    assert doc.raw_content == "Plain text content here."
    assert doc.source_type == "text"
    assert doc.metadata["file_name"] == "sample.txt"


def test_markdown_loader(tmp_path: Path) -> None:
    """Verify loading markdown document preserves content and sets source_type."""
    md_file = tmp_path / "sample.md"
    md_file.write_text("# Title\n\nParagraph text.", encoding="utf-8")

    loader = get_loader_for_path(md_file)
    assert isinstance(loader, TextLoader)

    doc = loader.load(md_file)
    assert doc.source_type == "markdown"
    assert "# Title" in doc.visible_text


def test_text_loader_missing_file(tmp_path: Path) -> None:
    """Verify error raised on non-existent file."""
    loader = TextLoader()
    with pytest.raises(FileNotFoundError):
        loader.load(tmp_path / "does_not_exist.txt")


def test_get_loader_unsupported_extension(tmp_path: Path) -> None:
    """Verify ValueError for unsupported extension."""
    unknown_file = tmp_path / "test.xyz_unsupported"
    with pytest.raises(ValueError, match="Unsupported file extension"):
        get_loader_for_path(unknown_file)
