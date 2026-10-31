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

    # Retrieval configuration
    top_k: int = Field(default=3, description="Number of nearest attack entries to retrieve")

    # Provider configurations (default to mock implementations)
    llm_provider: str = Field(default="mock", description="LLM provider name: 'mock', etc.")
    embedder_provider: str = Field(
        default="mock", description="Embedding provider name: 'mock', etc."
    )
    embedding_dimension: int = Field(default=128, description="Dimensionality of embeddings")

    # LiteLLM proxy (OpenAI-compatible) credentials and model names
    llm_base_url: str | None = Field(default=None, description="Base URL of the LiteLLM proxy")
    llm_api_key: SecretStr | None = Field(default=None, description="API key for the LiteLLM proxy")
    llm_model: str | None = Field(default=None, description="Chat model name exposed by the proxy")
    embedding_model: str | None = Field(
        default=None, description="Embedding model name exposed by the proxy"
    )
    report_language: str = Field(
        default="es",
        description="Language for free-text report fields ('es', 'en', 'pt' or a language name)",
    )
    llm_temperature: float = Field(default=0.0, description="Sampling temperature")
    llm_max_output_tokens: int = Field(default=800, description="Max output tokens per call")
    llm_timeout_s: float = Field(default=60.0, description="Request timeout in seconds")

    # Spending control (the team budget on the proxy is USD 10 in total)
    llm_cache_enabled: bool = Field(
        default=True, description="Reuse identical model calls from the on-disk cache"
    )
    budget_limit_usd: float = Field(
        default=8.0, description="Local spending cap; paid calls stop once it is reached"
    )

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
