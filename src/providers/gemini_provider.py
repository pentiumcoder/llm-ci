"""Google Gemini provider implementation."""

import logging
import time

import google.generativeai as genai

from src.config import Settings
from src.providers.base import BaseProvider
from src.providers.factory import ProviderError

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    """Google Gemini LLM provider using the generative AI API."""

    def __init__(self, settings: Settings) -> None:
        """Initialise with settings containing the API key."""
        genai.configure(api_key=settings.gemini_api_key)

    @property
    def name(self) -> str:
        """Return provider name."""
        return "gemini"

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
    ) -> dict:
        """Make a Gemini generation call with system + user prompt combined."""
        try:
            combined = f"{system_prompt}\n\n{user_message}"
            gemini_model = genai.GenerativeModel(model)

            start = time.perf_counter()
            response = await gemini_model.generate_content_async(combined)
            latency_ms = int((time.perf_counter() - start) * 1000)

            text = response.text or ""
            usage = response.usage_metadata

            return {
                "text": text,
                "input_tokens": usage.prompt_token_count if usage else 0,
                "output_tokens": usage.candidates_token_count if usage else 0,
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            logger.error("Gemini API call failed: %s", exc)
            raise ProviderError(f"Gemini API call failed: {exc}") from exc
