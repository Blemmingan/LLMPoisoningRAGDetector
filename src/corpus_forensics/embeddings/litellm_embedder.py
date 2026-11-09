"""Embedding client for the course LiteLLM proxy (OpenAI-compatible API).

Every text is cached on disk by (model, text), so re-ingesting the knowledge base or
re-running an evaluation never pays twice for the same embedding. Each paid batch is
recorded in the usage ledger and checked against the spending cap.
"""

from openai import OpenAI

from corpus_forensics.config import Settings, get_settings
from corpus_forensics.embeddings.base import Embedder
from corpus_forensics.llm.cache import ResponseCache
from corpus_forensics.llm.litellm_client import (
    build_openai_client,
    build_usage_tracker,
    call_with_retries,
    cost_from_headers,
)
from corpus_forensics.llm.usage import UsageRecord, UsageTracker, estimate_cost

KNOWN_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class LiteLLMEmbedder(Embedder):
    """Real embeddings through the LiteLLM proxy, with caching and cost accounting."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: OpenAI | None = None,
        tracker: UsageTracker | None = None,
        cache: ResponseCache | None = None,
        batch_size: int = 100,
        **_: object,
    ) -> None:
        self.settings = settings or get_settings()
        if not self.settings.embedding_model:
            raise ValueError("Embedding model not configured: set CF_EMBEDDING_MODEL in .env")
        self.model = self.settings.embedding_model
        self._client = client or build_openai_client(self.settings)
        self.tracker = tracker or build_usage_tracker(self.settings)
        self.cache = cache or ResponseCache(
            self.settings.cache_dir / "embeddings", enabled=self.settings.llm_cache_enabled
        )
        self.batch_size = batch_size
        name = self.model.split("/")[-1]
        self._dimension = KNOWN_DIMENSIONS.get(name, self.settings.embedding_dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_mock(self) -> bool:
        return False

    def _key(self, text: str) -> str:
        return ResponseCache.make_key(kind="embedding", model=self.model, text=text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, paying only for the ones not cached yet."""
        vectors: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for i, text in enumerate(texts):
            cached = self.cache.get(self._key(text))
            if cached is not None:
                vectors[i] = cached
            else:
                missing.append(i)

        if len(missing) < len(texts):
            self.tracker.record(
                UsageRecord(kind="embedding", model=self.model, cached=True, cost_source="cache")
            )

        for start in range(0, len(missing), self.batch_size):
            batch_idx = missing[start : start + self.batch_size]
            batch = [texts[i] or " " for i in batch_idx]
            self.tracker.check_budget()
            raw = call_with_retries(
                lambda batch=batch: self._client.embeddings.with_raw_response.create(  # type: ignore[misc]
                    model=self.model, input=batch
                ),
                max_retries=self.settings.llm_max_retries,
                backoff_s=self.settings.llm_retry_backoff_s,
            )
            response = raw.parse()
            tokens = response.usage.prompt_tokens if response.usage else 0
            proxy_cost = cost_from_headers(raw.headers)
            self.tracker.record(
                UsageRecord(
                    kind="embedding",
                    model=self.model,
                    prompt_tokens=tokens,
                    cost_usd=proxy_cost
                    if proxy_cost is not None
                    else estimate_cost(self.model, tokens, 0),
                    cost_source="proxy" if proxy_cost is not None else "estimate",
                )
            )
            for item in response.data:
                i = batch_idx[item.index]
                vectors[i] = list(item.embedding)
                self.cache.set(self._key(texts[i]), vectors[i])

        return [v if v is not None else [] for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        return self.embed_documents([text])[0]
