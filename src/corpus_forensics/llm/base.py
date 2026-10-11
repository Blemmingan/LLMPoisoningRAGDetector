"""Abstract base interface for LLM client providers."""

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


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
