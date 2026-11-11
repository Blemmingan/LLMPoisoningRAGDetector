"""Retriever for querying similar attack payloads and literature context from vector stores."""

from corpus_forensics.analysis.advanced_retrieval import (
    BM25Index,
    CrossEncoderReranker,
    cosine,
    reciprocal_rank_fusion,
    rewrite_query,
)
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
        query_rewrite: bool = False,
        query_rewrite_max_sentences: int = 3,
        hybrid_search: bool = False,
        rrf_k: int = 60,
        reranker: CrossEncoderReranker | None = None,
        rerank_candidates: int = 20,
    ) -> None:
        """Initialize retriever.

        Args:
            vector_store: Vector database containing known attacks.
            embedder: Embedding model instance.
            top_k: Number of nearest attack entries to retrieve.
            literature_store: Optional Chroma store containing research literature passages.
            literature_top_k: Number of nearest literature passages to retrieve.
            use_normalized_text: Search with the chunk's defensive normalization.
            query_rewrite: Search with the chunk's most suspicious sentences.
            query_rewrite_max_sentences: Sentences searched separately per chunk.
            hybrid_search: Fuse BM25 with the embedding search.
            rrf_k: Reciprocal Rank Fusion constant.
            reranker: Cross-encoder that reorders the candidate pool (None disables it).
            rerank_candidates: Candidate pool size for fusion and reranking.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.literature_store = literature_store
        self.literature_top_k = literature_top_k
        self.use_normalized_text = use_normalized_text
        self.query_rewrite = query_rewrite
        self.query_rewrite_max_sentences = query_rewrite_max_sentences
        self.hybrid_search = hybrid_search
        self.rrf_k = rrf_k
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates
        self._bm25: BM25Index | None = None

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
        if self.top_k <= 0 or not chunk.text.strip() or self.vector_store.count() == 0:
            return []  # top_k <= 0 disables the examples step (ablation: LLM alone)

        if query_embedding is None:
            query_embedding = self.embedder.embed_query(self.query_text(chunk))
        if not (self.query_rewrite or self.hybrid_search or self.reranker is not None):
            return self.vector_store.query(query_embedding=query_embedding, top_k=self.top_k)
        return self._advanced_retrieve(chunk, query_embedding)

    def _bm25_index(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(self.vector_store.get_all())
        return self._bm25

    def _advanced_retrieve(
        self, chunk: Chunk, chunk_embedding: list[float]
    ) -> list[RetrievedAttack]:
        """Rewriting -> dense (+ BM25) candidates fused with RRF -> optional reranking.

        Similarity scores of the final examples are always the cosine between the chunk
        embedding and the stored entry, so they stay comparable across configurations.
        """
        base_query = self.query_text(chunk)
        # the whole chunk is always searched; rewriting adds its suspicious sentences as
        # extra queries (fused with RRF), so focusing never loses the chunk's context
        focused = (
            rewrite_query(base_query, self.query_rewrite_max_sentences)
            if self.query_rewrite
            else []
        )
        queries = [base_query, *[q for q in focused if q != base_query]]
        pool = max(self.top_k, self.rerank_candidates)
        entries: dict[str, RetrievedAttack] = {}
        rankings: list[list[str]] = []
        for query in queries:
            embedding = chunk_embedding if query == base_query else self.embedder.embed_query(query)
            dense = self.vector_store.query(query_embedding=embedding, top_k=pool)
            entries.update({a.attack_id: a for a in dense})
            rankings.append([a.attack_id for a in dense])
            if self.hybrid_search:
                lexical = self._bm25_index().search(query, pool)
                entries.update({a.attack_id: a for a, _ in lexical if a.attack_id not in entries})
                rankings.append([a.attack_id for a, _ in lexical])
        candidates = (
            rankings[0] if len(rankings) == 1 else reciprocal_rank_fusion(rankings, self.rrf_k)
        )[:pool]

        rerank_scores: dict[str, float] = {}
        if self.reranker is not None and candidates:
            # the reranker reads the focused sentences when there are any
            focus = " ".join(queries[1:]) or base_query
            scores = self.reranker.score(focus, [entries[c].text for c in candidates])
            rerank_scores = dict(zip(candidates, scores, strict=True))
            candidates = sorted(candidates, key=lambda c: -rerank_scores[c])

        final = candidates[: self.top_k]
        stored = self.vector_store.get_embeddings(final)
        results: list[RetrievedAttack] = []
        for attack_id in final:
            attack = entries[attack_id]
            similarity = (
                cosine(chunk_embedding, stored[attack_id])
                if attack_id in stored
                else attack.similarity_score
            )
            metadata = dict(attack.metadata)
            if attack_id in rerank_scores:
                metadata["rerank_score"] = round(rerank_scores[attack_id], 4)
            results.append(
                attack.model_copy(
                    update={
                        "similarity_score": round(max(0.0, min(1.0, similarity)), 4),
                        "metadata": metadata,
                    }
                )
            )
        return results

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
