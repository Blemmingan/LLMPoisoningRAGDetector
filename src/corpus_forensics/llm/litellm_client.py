"""LLM client for the course LiteLLM proxy (OpenAI-compatible API).

Uses the official `openai` SDK pointed at the proxy base URL and OpenAI structured
outputs, so responses are parsed and validated against a Pydantic schema.
Every call goes through the on-disk cache and the usage tracker.
"""

import time
from collections.abc import Callable
from typing import Any, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    OpenAIError,
)
from pydantic import BaseModel

from corpus_forensics.config import Settings, get_settings
from corpus_forensics.llm.base import LLMClient, LLMError, ProviderQuotaError
from corpus_forensics.llm.cache import ResponseCache
from corpus_forensics.llm.usage import UsageRecord, UsageTracker, estimate_cost

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")

COST_HEADER = "x-litellm-response-cost"
QUOTA_MARKERS = ("insufficient_quota", "credit_balance_exhausted", "no credits remaining")


def is_quota_error(exc: BaseException) -> bool:
    """True if the provider rejected the call for lack of credit (not a transient limit)."""
    if not isinstance(exc, APIStatusError) or exc.status_code != 429:
        return False
    text = f"{exc} {getattr(exc, 'body', '')}".lower()
    return any(marker in text for marker in QUOTA_MARKERS)


def is_transient_error(exc: BaseException) -> bool:
    """Network problems, timeouts, real rate limits and 5xx errors are worth retrying."""
    if isinstance(exc, APIConnectionError | APITimeoutError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500 or (exc.status_code == 429 and not is_quota_error(exc))
    return False


def call_with_retries(
    fn: Callable[[], R],
    max_retries: int = 2,
    backoff_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> R:
    """Run `fn`, retrying only transient errors.

    A lack-of-credit error raises ProviderQuotaError immediately, with no retry: retrying
    only multiplies failed requests against the proxy.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except OpenAIError as exc:
            if is_quota_error(exc):
                raise ProviderQuotaError(
                    "The provider behind the LiteLLM proxy has no credit left "
                    "(insufficient_quota). Stopping the run; switch CF_LLM_PROVIDER / "
                    "CF_EMBEDDER_PROVIDER to 'mock' or 'local' meanwhile."
                ) from exc
            if attempt >= max_retries or not is_transient_error(exc):
                raise LLMError(f"LiteLLM call failed: {exc}") from exc
            sleep(backoff_s * (2**attempt))
            attempt += 1


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
        max_retries=0,  # retries are handled by call_with_retries (never on lack of credit)
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
        raw = call_with_retries(
            lambda: self._client.chat.completions.with_raw_response.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=schema,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_output_tokens,
            ),
            max_retries=self.settings.llm_max_retries,
            backoff_s=self.settings.llm_retry_backoff_s,
        )
        completion = raw.parse()

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
