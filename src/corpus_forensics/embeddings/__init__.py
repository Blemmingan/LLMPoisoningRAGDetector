"""Embeddings module."""

from corpus_forensics.embeddings.base import Embedder
from corpus_forensics.embeddings.factory import get_embedder, register_embedder
from corpus_forensics.embeddings.mock import MockEmbedder

__all__ = [
    "Embedder",
    "MockEmbedder",
    "get_embedder",
    "register_embedder",
]
