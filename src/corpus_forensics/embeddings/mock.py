"""# MOCK deterministic hash-based embedding implementation."""

import hashlib
import math
import re

from corpus_forensics.embeddings.base import Embedder


class MockEmbedder(Embedder):
    """# MOCK Deterministic hash-based vector embedder.

    Produces normalized float vectors of fixed dimension using sha256 hashing.
    Enables reproducible cosine similarity without external ML dependencies or network calls.
    """

    def __init__(self, dimension: int = 128) -> None:
        """Initialize mock embedder.

        Args:
            dimension: Dimensionality of the generated vector.
        """
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def is_mock(self) -> bool:
        return True

    def _hash_text_to_vector(self, text: str) -> list[float]:
        """Convert text into a deterministic, unit-normalized float vector."""
        vec = [0.0] * self._dim

        # Split text into lowercase tokens
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            tokens = ["<empty>"]

        # Accumulate token hashes to simulate semantic bag-of-words
        for token in tokens:
            token_bytes = token.encode("utf-8")
            h = hashlib.sha256(token_bytes).digest()
            for i in range(self._dim):
                byte_val = h[i % len(h)]
                # map byte 0-255 to range [-1.0, 1.0]
                val = (byte_val / 127.5) - 1.0
                vec[i] += val

        # Add a document-level sha256 hash component to break symmetries
        doc_h = hashlib.sha256(text.encode("utf-8")).digest()
        for i in range(self._dim):
            doc_val = (doc_h[i % len(doc_h)] / 127.5) - 1.0
            vec[i] += 0.5 * doc_val

        # Normalize to unit length (L2 norm)
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            vec = [x / norm for x in vec]
        else:
            vec[0] = 1.0

        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """# MOCK Generate deterministic vectors for a batch of texts."""
        return [self._hash_text_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """# MOCK Generate deterministic vector for a single query."""
        return self._hash_text_to_vector(text)
