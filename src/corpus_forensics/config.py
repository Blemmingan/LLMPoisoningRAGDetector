"""Configuration module using pydantic-settings.

Defines runtime configuration, path defaults, chunking parameters,
and provider options for Corpus Forensic Analyst.
"""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable override support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CF_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base workspace paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data"
    )
    kb_dir: Path = Field(
        default_factory=lambda: (
            Path(__file__).resolve().parent.parent.parent / "data" / "knowledge_base"
        )
    )
    eval_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "eval"
    )
    vectorstore_dir: Path = Field(
        default_factory=lambda: (
            Path(__file__).resolve().parent.parent.parent / "data" / "vectorstore"
        )
    )
    samples_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "samples"
    )
    output_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "reports"
    )
    cache_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "cache"
    )
    usage_ledger_path: Path = Field(
        default_factory=lambda: (
            Path(__file__).resolve().parent.parent.parent / "data" / "usage" / "ledger.jsonl"
        )
    )

    # Chunking configuration
    chunk_size: int = Field(default=500, description="Chunk size in characters or tokens")
    chunk_overlap: int = Field(default=100, description="Chunk overlap in characters or tokens")
    chunk_unit: str = Field(default="char", description="Unit for chunking: 'char' or 'token'")

    # Preprocessing: search with the defensive normalization of each chunk (ablation flag)
    normalize_for_retrieval: bool = Field(
        default=True,
        description="Embed the normalized chunk (decoded, de-obfuscated) instead of the raw text",
    )

    # Retrieval configuration
    top_k: int = Field(default=3, description="Number of nearest attack entries to retrieve")
    literature_top_k: int = Field(
        default=2,
        description="Literature passages added to the analyst prompt (0 disables the step)",
    )

    # Analyst prompt: file name in analysis/prompts/ (without .txt) or a path (ablation)
    analyst_prompt: str = Field(
        default="analyst_prompt",
        description="Analyst prompt template: name in analysis/prompts/ or path to a .txt file",
    )
    # Verifier: lower the confidence of findings whose fragment is not in the document
    verifier_penalize: bool = Field(
        default=True,
        description="Multiply the confidence of unverified findings by 0.2 (anti-hallucination)",
    )

    # Document verdict: a document is malicious when at least one finding passes the policy
    verdict_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum finding confidence to flag a document"
    )
    verdict_require_verified: bool = Field(
        default=True,
        description="Only count findings whose fragment was found verbatim in the document",
    )
    verdict_min_severity: str = Field(
        default="baja",
        description="Minimum finding severity that counts: 'baja', 'media', 'alta' or 'critica'",
    )

    # Advanced retrieval of known attacks (no LLM): rewriting -> hybrid -> reranking.
    # All on by default ("rag_avanzado", the best configuration in the ablation).
    query_rewrite: bool = Field(
        default=True, description="Search with the most suspicious sentences of the chunk"
    )
    query_rewrite_max_sentences: int = Field(
        default=3, description="Suspicious sentences searched separately per chunk"
    )
    hybrid_search: bool = Field(
        default=True, description="Fuse BM25 with the embedding search (Reciprocal Rank Fusion)"
    )
    rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion constant")
    rerank: bool = Field(default=True, description="Rerank candidates with a cross-encoder")
    rerank_model: str = Field(
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        description="Cross-encoder model name or local path",
    )
    rerank_candidates: int = Field(
        default=20, description="Candidate pool reranked (or fused) before keeping top_k"
    )

    # Provider configurations (default to mock implementations)
    llm_provider: str = Field(default="mock", description="LLM provider name: 'mock', etc.")
    embedder_provider: str = Field(
        default="mock", description="Embedding provider name: 'mock', etc."
    )
    embedding_dimension: int = Field(default=128, description="Dimensionality of embeddings")

    # Local embeddings (CF_EMBEDDER_PROVIDER=local)
    local_embedding_model: str = Field(
        default="intfloat/multilingual-e5-small",
        description="sentence-transformers model name or local path",
    )
    local_embedding_query_prefix: str = Field(default="query: ", description="E5 query prefix")
    local_embedding_passage_prefix: str = Field(
        default="passage: ", description="E5 passage prefix"
    )

    # LiteLLM proxy (OpenAI-compatible) credentials and model names
    llm_base_url: str | None = Field(default=None, description="Base URL of the LiteLLM proxy")
    llm_api_key: SecretStr | None = Field(default=None, description="API key for the LiteLLM proxy")
    llm_model: str | None = Field(default=None, description="Chat model name exposed by the proxy")
    embedding_model: str | None = Field(
        default=None, description="Embedding model name exposed by the proxy"
    )
    # Literature collection (papers and technical reports)
    lit_chunk_tokens: int = Field(default=350, description="Target tokens per literature chunk")
    lit_chunk_overlap: int = Field(default=50, description="Overlap tokens between chunks")
    lit_priorities: str = Field(
        default="alta", description="Comma-separated priorities of sources to use"
    )

    report_language: str = Field(
        default="es",
        description="Language for free-text report fields ('es', 'en', 'pt' or a language name)",
    )
    hf_token: SecretStr | None = Field(
        default=None, description="Hugging Face read token (gated HackAPrompt dataset)"
    )
    llm_temperature: float = Field(default=0.0, description="Sampling temperature")
    llm_max_output_tokens: int = Field(default=800, description="Max output tokens per call")
    llm_timeout_s: float = Field(default=60.0, description="Request timeout in seconds")
    llm_max_retries: int = Field(
        default=2, description="Retries for transient errors (never for lack of credit)"
    )
    llm_retry_backoff_s: float = Field(default=2.0, description="Initial retry backoff")

    # Spending control (the team budget on the proxy is USD 10 in total)
    llm_cache_enabled: bool = Field(
        default=True, description="Reuse identical model calls from the on-disk cache"
    )
    budget_limit_usd: float = Field(
        default=8.0, description="Local spending cap; paid calls stop once it is reached"
    )

    @property
    def kb_processed_dir(self) -> Path:
        """Directory for processed knowledge-base artifacts (markdown, chunks)."""
        return self.kb_dir / "processed"

    @property
    def literature_sources_path(self) -> Path:
        """Versioned list of literature sources."""
        return self.kb_dir / "literature_sources.yaml"

    @property
    def kb_raw_dir(self) -> Path:
        """Directory where `kb download` stores the raw files of each source."""
        return self.kb_dir / "raw"

    @property
    def llm_is_mock(self) -> bool:
        """Indicate whether the active LLM provider is a mock."""
        return self.llm_provider.lower() == "mock"

    @property
    def embedder_is_mock(self) -> bool:
        """Indicate whether the active embedder provider is a mock."""
        return self.embedder_provider.lower() == "mock"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Retrieve singleton configuration instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings(new_settings: Settings | None = None) -> None:
    """Reset configuration instance (useful for testing)."""
    global _settings
    _settings = new_settings
