"""Configurable overlapping document chunker preserving original character offsets."""

import re
from typing import Literal

from corpus_forensics.preprocessing.normalizer import normalize_and_annotate
from corpus_forensics.schemas import Chunk, Document


class Chunker:
    """Chunks documents with configurable window size, overlap, and unit.

    Accurately tracks start and end character offsets into the original document.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        unit: Literal["char", "token"] = "char",
    ) -> None:
        """Initialize Chunker.

        Args:
            chunk_size: Size of each chunk in characters or tokens.
            chunk_overlap: Overlapping characters or tokens between consecutive chunks.
            unit: 'char' or 'token'.

        Raises:
            ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >= chunk_size.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.unit = unit

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a document into overlapping chunks preserving original character offsets.

        Args:
            document: Ingestion Document containing visible_text (and optionally hidden_content).

        Returns:
            List of Chunk objects with start/end character offsets and annotations.
        """
        # Visible text plus the hidden channel (if any), so hidden payloads are analyzed too.
        # The verifier resolves each finding back to its channel and channel-relative offsets.
        text_to_chunk = document.analyzed_text

        if not text_to_chunk:
            return []

        if self.unit == "char":
            return self._chunk_by_char(text_to_chunk, document.id)
        else:
            return self._chunk_by_token(text_to_chunk, document.id)

    def _chunk_by_char(self, text: str, doc_id: str) -> list[Chunk]:
        """Slice text into overlapping character chunks."""
        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        total_len = len(text)

        start = 0
        idx = 0
        while start < total_len:
            end = min(start + self.chunk_size, total_len)
            chunk_text = text[start:end]

            _, annotations = normalize_and_annotate(chunk_text)

            chunks.append(
                Chunk(
                    id=f"{doc_id}_chunk_{idx}",
                    document_id=doc_id,
                    text=chunk_text,
                    start=start,
                    end=end,
                    index=idx,
                    annotations=annotations,
                )
            )
            idx += 1
            if end >= total_len:
                break
            start += step

        return chunks

    def _chunk_by_token(self, text: str, doc_id: str) -> list[Chunk]:
        """Slice text into overlapping token chunks while preserving character offsets."""
        # Find all word-like tokens with their character spans
        token_matches = list(re.finditer(r"\S+", text))
        if not token_matches:
            return []

        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        total_tokens = len(token_matches)

        token_start = 0
        idx = 0
        while token_start < total_tokens:
            token_end = min(token_start + self.chunk_size, total_tokens)

            start_char = token_matches[token_start].start()
            end_char = token_matches[token_end - 1].end()
            chunk_text = text[start_char:end_char]

            _, annotations = normalize_and_annotate(chunk_text)

            chunks.append(
                Chunk(
                    id=f"{doc_id}_chunk_{idx}",
                    document_id=doc_id,
                    text=chunk_text,
                    start=start_char,
                    end=end_char,
                    index=idx,
                    annotations=annotations,
                )
            )
            idx += 1
            if token_end >= total_tokens:
                break
            token_start += step

        return chunks
