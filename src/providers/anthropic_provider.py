"""Anthropic provider implementation."""

import logging
import time

from anthropic import AsyncAnthropic

from src.config import Settings
from src.providers.base import BaseProvider
from src.providers.factory import ProviderError

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """Anthropic LLM provider using the messages API."""

    def __init__(self, settings: Settings) -> None:
        """Initialise with settings containing the API key."""
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    @property
    def name(self) -> str:
        """Return provider name."""
        return "anthropic"

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
    ) -> dict:
        """Make an async Anthropic messages API call."""
        try:
            start = time.perf_counter()
            response = await self._client.messages.create(
                model=model,
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message},
                ],
            )
            latency_ms = int((time.perf_counter() - start) * 1000)

            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text

            return {
                "text": text,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            logger.error("Anthropic API call failed: %s", exc)
            raise ProviderError(f"Anthropic API call failed: {exc}") from exc
