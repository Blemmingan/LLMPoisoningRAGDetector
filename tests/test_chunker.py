"""Tests for overlapping chunker and character offset tracking."""

import pytest

from corpus_forensics.preprocessing.chunker import Chunker
from corpus_forensics.schemas import Document


def test_chunker_validation() -> None:
    """Test chunker parameter boundary validation."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        Chunker(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        Chunker(chunk_size=100, chunk_overlap=-5)

    with pytest.raises(ValueError, match="must be strictly less than"):
        Chunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="must be strictly less than"):
        Chunker(chunk_size=100, chunk_overlap=150)


def test_chunker_char_offsets_and_overlap(sample_document: Document) -> None:
    """Verify that every chunk text precisely matches the original document slice."""
    chunk_size = 50
    overlap = 15
    chunker = Chunker(chunk_size=chunk_size, chunk_overlap=overlap, unit="char")
    chunks = chunker.chunk_document(sample_document)

    assert len(chunks) > 1

    source_text = sample_document.visible_text

    for i, c in enumerate(chunks):
        assert c.index == i
        # Exact character slice test:
        assert source_text[c.start : c.end] == c.text
        assert len(c.text) <= chunk_size

        if i > 0:
            prev = chunks[i - 1]
            # Overlap verification: step is chunk_size - overlap
            step = chunk_size - overlap
            assert c.start == prev.start + step


def test_chunker_empty_document() -> None:
    """Verify handling of empty text."""
    doc = Document(
        id="empty", raw_content="", visible_text="", hidden_content="", source_type="text"
    )
    chunker = Chunker(chunk_size=100, chunk_overlap=20)
    assert chunker.chunk_document(doc) == []


def test_chunker_short_document() -> None:
    """Verify text shorter than chunk size produces exactly one chunk."""
    doc = Document(
        id="short", raw_content="Hello world", visible_text="Hello world", source_type="text"
    )
    chunker = Chunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].start == 0
    assert chunks[0].end == len("Hello world")
    assert chunks[0].text == "Hello world"


def test_chunker_token_offsets() -> None:
    """Verify token-based chunking preserves precise character offsets."""
    text = "The quick brown fox jumps over the lazy dog repeatedly."
    doc = Document(id="token_doc", raw_content=text, visible_text=text, source_type="text")

    chunker = Chunker(chunk_size=4, chunk_overlap=1, unit="token")
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    for c in chunks:
        # Crucial check: slice of original document matches chunk text exactly
        assert text[c.start : c.end] == c.text
