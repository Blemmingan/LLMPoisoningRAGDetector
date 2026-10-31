"""LLM client for the course LiteLLM proxy (OpenAI-compatible API).

Uses the official `openai` SDK pointed at the proxy base URL and OpenAI structured
outputs, so responses are parsed and validated against a Pydantic schema.
Every call goes through the on-disk cache and the usage tracker.
"""

from typing import Any, TypeVar

from openai import OpenAI, OpenAIError
from pydantic import BaseModel

from corpus_forensics.config import Settings, get_settings
from corpus_forensics.llm.base import LLMClient, LLMError
from corpus_forensics.llm.cache import ResponseCache
from corpus_forensics.llm.usage import UsageRecord, UsageTracker, estimate_cost

T = TypeVar("T", bound=BaseModel)

COST_HEADER = "x-litellm-response-cost"


def build_usage_tracker(settings: Settings) -> UsageTracker:
    """Create the usage tracker configured in settings."""
    return UsageTracker(settings.usage_ledger_path, settings.budget_limit_usd)


def build_openai_client(settings: Settings) -> OpenAI:
    """Create an OpenAI SDK client pointed at the LiteLLM proxy."""
    if not settings.llm_base_url or settings.llm_api_key is None:
        raise ValueError(
            "LiteLLM proxy not configured: set CF_LLM_BASE_URL and CF_LLM_API_KEY in .env"
        )
    return OpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        timeout=settings.llm_timeout_s,
        max_retries=2,
    )


def cost_from_headers(headers: Any) -> float | None:
    """Read the real call cost reported by the LiteLLM proxy, if present."""
    raw = headers.get(COST_HEADER) if headers is not None else None
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


class LiteLLMClient(LLMClient):
    """Structured-output chat client for the LiteLLM proxy."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: OpenAI | None = None,
        tracker: UsageTracker | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if not self.settings.llm_model:
            raise ValueError("Chat model not configured: set CF_LLM_MODEL in .env")
        self.model = self.settings.llm_model
        self._client = client or build_openai_client(self.settings)
        self.tracker = tracker or build_usage_tracker(self.settings)
        self.cache = cache or ResponseCache(
            self.settings.cache_dir / "llm", enabled=self.settings.llm_cache_enabled
        )

    @property
    def provider_name(self) -> str:
        return "litellm"

    @property
    def is_mock(self) -> bool:
        return False

    def generate_structured(self, prompt: str, schema: type[T]) -> T:
        """Call the chat model and parse its answer into `schema`."""
        key = ResponseCache.make_key(
            kind="chat",
            model=self.model,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_output_tokens,
            schema=schema.model_json_schema(),
            prompt=prompt,
        )

        cached = self.cache.get(key)
        if cached is not None:
            self.tracker.record(
                UsageRecord(kind="chat", model=self.model, cached=True, cost_source="cache")
            )
            return schema.model_validate(cached)

        self.tracker.check_budget()
        try:
            raw = self._client.chat.completions.with_raw_response.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=schema,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_output_tokens,
            )
            completion = raw.parse()
        except OpenAIError as exc:
            raise LLMError(f"LiteLLM call failed: {exc}") from exc

        usage = completion.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        proxy_cost = cost_from_headers(raw.headers)
        self.tracker.record(
            UsageRecord(
                kind="chat",
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=(
                    proxy_cost
                    if proxy_cost is not None
                    else estimate_cost(self.model, prompt_tokens, completion_tokens)
                ),
                cost_source="proxy" if proxy_cost is not None else "estimate",
            )
        )

        message = completion.choices[0].message
        if message.refusal:
            raise LLMError(f"Model refused to answer: {message.refusal}")
        if message.parsed is None:
            raise LLMError("Model returned no parseable structured output")

        parsed = schema.model_validate(message.parsed.model_dump())
        self.cache.set(key, parsed.model_dump(mode="json"))
        return parsed
