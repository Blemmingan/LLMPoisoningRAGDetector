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
        use_normalized_text: bool = True,
    ) -> None:
        """Initialize retriever.

        Args:
            vector_store: Vector database containing known attacks.
            embedder: Embedding model instance.
            top_k: Number of nearest attack entries to retrieve.
            literature_store: Optional Chroma store containing research literature passages.
            literature_top_k: Number of nearest literature passages to retrieve.
            use_normalized_text: Search with the chunk's defensive normalization.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.literature_store = literature_store
        self.literature_top_k = literature_top_k
        self.use_normalized_text = use_normalized_text

    def query_text(self, chunk: Chunk) -> str:
        """Text used to search: the normalized view when enabled and available."""
        if self.use_normalized_text and chunk.normalized_text.strip():
            return chunk.normalized_text
        return chunk.text

    def embed_chunk(self, chunk: Chunk) -> list[float] | None:
        """Embed a chunk once so both retrieval steps can reuse the same vector."""
        if not chunk.text.strip():
            return None
        return self.embedder.embed_query(self.query_text(chunk))

    def retrieve_for_chunk(
        self, chunk: Chunk, query_embedding: list[float] | None = None
    ) -> list[RetrievedAttack]:
        """Query the vector store for the known attack examples closest to a chunk.

        Args:
            chunk: Text chunk being analyzed.
            query_embedding: Pre-computed chunk embedding (computed here if omitted).

        Returns:
            List of RetrievedAttack items.
        """
        if not chunk.text.strip() or self.vector_store.count() == 0:
            return []

        if query_embedding is None:
            query_embedding = self.embedder.embed_query(self.query_text(chunk))
        return self.vector_store.query(query_embedding=query_embedding, top_k=self.top_k)

    def retrieve_literature_for_chunk(
        self, chunk: Chunk, query_embedding: list[float] | None = None
    ) -> list[LiteraturePassage]:
        """Query the literature store for research passages relevant to a chunk.

        Passages flagged `contiene_payload` (papers quoting literal attack strings) are
        skipped: the knowledge base must not hand injection text to the analyst.
        `literature_top_k <= 0` disables this step (e.g. for the ablation study).

        Args:
            chunk: Text chunk being analyzed.
            query_embedding: Pre-computed chunk embedding (computed here if omitted).

        Returns:
            Up to `literature_top_k` LiteraturePassage items without literal payloads.
        """
        if (
            self.literature_top_k <= 0
            or not chunk.text.strip()
            or self.literature_store is None
            or self.literature_store.count() == 0
        ):
            return []

        if query_embedding is None:
            query_embedding = self.embedder.embed_query(self.query_text(chunk))
        # Over-fetch so that dropping payload passages still leaves enough results.
        candidates = self.literature_store.query(
            query_embedding=query_embedding, top_k=self.literature_top_k * 3 + 2
        )
        safe = [p for p in candidates if not p.metadata.get("contiene_payload")]
        return safe[: self.literature_top_k]
