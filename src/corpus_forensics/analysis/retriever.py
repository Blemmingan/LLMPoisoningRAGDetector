"""Retriever for querying similar attack payloads and literature context from vector stores."""

from corpus_forensics.embeddings.base import Embedder
from corpus_forensics.schemas import Chunk, RetrievedAttack
from corpus_forensics.vectorstore.base import VectorStore
from corpus_forensics.vectorstore.literature_store import LiteraturePassage, LiteratureStore


class AttackRetriever:
    """Retrieves top-k known attack entries and literature passages similar to a given chunk."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        top_k: int = 3,
        literature_store: LiteratureStore | None = None,
        literature_top_k: int = 2,
    ) -> None:
        """Initialize retriever.

        Args:
            vector_store: Vector database containing known attacks.
            embedder: Embedding model instance.
            top_k: Number of nearest attack entries to retrieve.
            literature_store: Optional Chroma store containing research literature passages.
            literature_top_k: Number of nearest literature passages to retrieve.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.literature_store = literature_store
        self.literature_top_k = literature_top_k

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

    def retrieve_literature_for_chunk(self, chunk: Chunk) -> list[LiteraturePassage]:
        """Embed chunk text and query literature store for relevant research passages.

        Args:
            chunk: Text chunk being analyzed.

        Returns:
            List of LiteraturePassage items.
        """
        if (
            not chunk.text.strip()
            or self.literature_store is None
            or self.literature_store.count() == 0
        ):
            return []

        query_embedding = self.embedder.embed_query(chunk.text)
        return self.literature_store.query(
            query_embedding=query_embedding, top_k=self.literature_top_k
        )
