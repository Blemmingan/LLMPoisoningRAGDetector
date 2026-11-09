"""Local embeddings with sentence-transformers (default: intfloat/multilingual-e5-small).

Free, offline after the first download, and multilingual (Spanish and English share the
same vector space). E5 models expect a prefix on every input: "query: " for what is being
searched and "passage: " for what is stored; vectors are L2-normalized for cosine search.
"""

from typing import Any

from corpus_forensics.config import Settings, get_settings
from corpus_forensics.embeddings.base import Embedder


class LocalEmbedder(Embedder):
    """sentence-transformers embedder running on the local CPU/GPU."""

    def __init__(
        self,
        settings: Settings | None = None,
        model: Any | None = None,
        batch_size: int = 32,
        **_: object,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = self.settings.local_embedding_model
        self.query_prefix = self.settings.local_embedding_query_prefix
        self.passage_prefix = self.settings.local_embedding_passage_prefix
        self.batch_size = batch_size
        self._model = model  # injected in tests; loaded lazily otherwise

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    @property
    def is_mock(self) -> bool:
        return False

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [[float(x) for x in v] for v in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed stored passages (attack examples, literature chunks)."""
        return self._encode([f"{self.passage_prefix}{t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        """Embed a search input (a document chunk under analysis, a finding)."""
        return self._encode([f"{self.query_prefix}{text}"])[0]
