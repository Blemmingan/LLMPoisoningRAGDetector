"""Retriever for querying similar attack payloads from vector store."""

from corpus_forensics.embeddings.base import Embedder
from corpus_forensics.schemas import Chunk, RetrievedAttack
from corpus_forensics.vectorstore.base import VectorStore


class AttackRetriever:
    """Retrieves top-k known attack entries similar to a given chunk."""

    def __init__(self, vector_store: VectorStore, embedder: Embedder, top_k: int = 3) -> None:
        """Initialize retriever.

        Args:
            vector_store: Vector database containing known attacks.
            embedder: Embedding model instance.
            top_k: Number of nearest attack entries to retrieve.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve_for_chunk(self, chunk: Chunk) -> list[RetrievedAttack]:
        """Embed chunk text and query vector store for closest known attack examples.

        Args:
            chunk: Text chunk being analyzed.

        Returns:
            List of RetrievedAttack items.
        """
        if not chunk.text.strip() or self.vector_store.count() == 0:
            return []

        query_embedding = self.embedder.embed_query(chunk.text)
        return self.vector_store.query(query_embedding=query_embedding, top_k=self.top_k)
