"""Abstract base interface for LLM client providers."""

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Raised when an LLM call fails or returns an unusable response."""


class BudgetExceededError(LLMError):
    """Raised before a paid call when the configured spending cap has been reached.

    Any subclass stops the whole run: the analyst re-raises it instead of skipping a chunk.
    """


class ProviderQuotaError(BudgetExceededError):
    """Raised when the provider behind the proxy has no credit left (HTTP 429
    `insufficient_quota`). Retrying cannot help, so the run stops at the first one."""


class LLMClient(ABC):
    """Abstract interface for LLM providers supporting structured output generation."""

    @abstractmethod
    def generate_structured(self, prompt: str, schema: type[T]) -> T:
        """Invoke LLM and parse response into the requested Pydantic schema.

        Args:
            prompt: Text prompt sent to the LLM.
            schema: Pydantic model class representing expected structured output.

        Returns:
            Validated instance of schema.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the LLM provider."""
        pass

    @property
    @abstractmethod
    def is_mock(self) -> bool:
        """Flag indicating if this client is a mock."""
        pass
