"""Google Gemini provider implementation using the google-genai SDK."""

import asyncio
import logging
import re
import time

from google import genai
from google.genai import errors

from src.config import Settings
from src.providers.base import BaseProvider
from src.providers.factory import ProviderError

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 8
_FALLBACK_RETRY_SECONDS = 65.0
_RETRY_DELAY_RE = re.compile(
    r"(?:seconds|retryDelay|retry in)[:'\s]*(\d+(?:\.\d+)?)"
)


class _RateLimiter:
    """Async sliding-window rate limiter to stay under provider RPM caps."""

    def __init__(self, max_calls: int, window_seconds: float) -> None:
        self._max_calls = max_calls
        self._window_seconds = window_seconds
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a call slot is available within the rate window."""
        while True:
            async with self._lock:
                now = time.monotonic()
                self._timestamps = [
                    ts for ts in self._timestamps
                    if now - ts < self._window_seconds
                ]
                if len(self._timestamps) < self._max_calls:
                    self._timestamps.append(now)
                    return
                wait = self._window_seconds - (now - self._timestamps[0])

            await asyncio.sleep(wait)


def _is_rate_limited(exc: Exception) -> bool:
    """Return True if the exception is a Gemini API rate-limit error."""
    if isinstance(exc, errors.APIError) and exc.code == 429:
        return True
    message = str(exc)
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def _retry_delay_seconds(exc: Exception) -> float:
    """Extract the API-suggested retry delay from a rate-limit error."""
    match = _RETRY_DELAY_RE.search(str(exc))
    if match:
        return float(match.group(1)) + 1.0
    return _FALLBACK_RETRY_SECONDS


class GeminiProvider(BaseProvider):
    """Google Gemini LLM provider using the google-genai SDK."""

    def __init__(self, settings: Settings) -> None:
        """Initialise with settings containing the API key."""
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._rate_limiter = _RateLimiter(
            max_calls=settings.gemini_requests_per_minute,
            window_seconds=60.0,
        )

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
        combined = f"{system_prompt}\n\n{user_message}"
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                await self._rate_limiter.acquire()

                start = time.perf_counter()
                response = await self._client.aio.models.generate_content(
                    model=model,
                    contents=combined,
                )
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
                last_exc = exc
                if _is_rate_limited(exc):
                    retry_seconds = _retry_delay_seconds(exc)
                    logger.warning(
                        "Gemini rate limit hit (attempt %d/%d); backing off %.1fs",
                        attempt,
                        _MAX_ATTEMPTS,
                        retry_seconds,
                    )
                    await asyncio.sleep(retry_seconds)
                    continue
                logger.error("Gemini API call failed: %s", exc)
                raise ProviderError(f"Gemini API call failed: {exc}") from exc

        logger.error("Gemini API call failed after %d attempts: %s", _MAX_ATTEMPTS, last_exc)
        raise ProviderError(f"Gemini API call failed: {last_exc}") from last_exc
