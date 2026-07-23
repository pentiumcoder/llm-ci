"""OpenAI provider implementation."""

import json
import logging
import time

from openai import AsyncOpenAI

from src.config import Settings
from src.providers.base import BaseProvider
from src.providers.factory import ProviderError

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI LLM provider using the chat completions API."""

    def __init__(self, settings: Settings) -> None:
        """Initialise with settings containing the API key."""
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    @property
    def name(self) -> str:
        """Return provider name."""
        return "openai"

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
    ) -> dict:
        """Make an async OpenAI chat completion call."""
        try:
            start = time.perf_counter()
            response = await self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            latency_ms = int((time.perf_counter() - start) * 1000)

            choice = response.choices[0]
            usage = response.usage

            return {
                "text": choice.message.content or "",
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            logger.error("OpenAI API call failed: %s", exc)
            raise ProviderError(f"OpenAI API call failed: {exc}") from exc
