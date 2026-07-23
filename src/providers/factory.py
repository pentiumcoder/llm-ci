"""Factory function for selecting the correct LLM provider."""

import logging

from src.config import Settings
from src.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Raised when an LLM provider API call fails."""


def get_provider(name: str, settings: Settings) -> BaseProvider:
    """Return the correct provider instance for a given name string."""
    if name == "openai":
        from src.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(settings)
    elif name == "anthropic":
        from src.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings)
    elif name == "gemini":
        from src.providers.gemini_provider import GeminiProvider

        return GeminiProvider(settings)
    else:
        raise ValueError(f"Unknown provider: {name!r}. Supported: openai, anthropic, gemini.")
