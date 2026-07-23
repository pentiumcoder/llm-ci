"""Abstract base class for all LLM providers."""

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
    ) -> dict:
        """Make a completion call. Return dict with keys: text, input_tokens, output_tokens, latency_ms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name string e.g. 'openai', 'anthropic', 'gemini'."""
