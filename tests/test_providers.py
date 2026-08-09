"""Tests for LLM providers, factory, and feature function."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import Settings
from src.cost import calculate_cost_aud
from src.feature import classify_email, load_prompt
from src.models import PromptConfig
from src.providers.base import BaseProvider
from src.providers.factory import ProviderError, get_provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings(**overrides) -> Settings:
    defaults = {
        "openai_api_key": "sk-test",
        "anthropic_api_key": "sk-ant-test",
        "gemini_api_key": "gemini-test",
        "provider": "openai",
        "model_under_test": "gpt-4o-mini",
    }
    defaults.update(overrides)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# BaseProvider
# ---------------------------------------------------------------------------

class TestBaseProvider:
    """Tests for BaseProvider ABC."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseProvider()

    def test_subclass_must_implement_complete(self):
        class Incomplete(BaseProvider):
            @property
            def name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_must_implement_name(self):
        class Incomplete(BaseProvider):
            async def complete(self, system_prompt, user_message, model):
                return {}

        with pytest.raises(TypeError):
            Incomplete()

    def test_valid_subclass(self):
        class Valid(BaseProvider):
            async def complete(self, system_prompt, user_message, model):
                return {"text": "ok", "input_tokens": 1, "output_tokens": 1, "latency_ms": 1}

            @property
            def name(self) -> str:
                return "valid"

        p = Valid()
        assert p.name == "valid"


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------

class TestOpenAIProvider:
    """Tests for OpenAIProvider with mocked API."""

    def _make_provider(self):
        from src.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(_settings())

    @pytest.mark.asyncio
    async def test_complete_returns_correct_schema(self):
        provider = self._make_provider()
        mock_message = SimpleNamespace(content="Hello")
        mock_usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        mock_choice = SimpleNamespace(message=mock_message)
        mock_response = SimpleNamespace(choices=[mock_choice], usage=mock_usage)

        provider._client = MagicMock()
        provider._client.chat.completions = MagicMock()
        provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.complete("system", "user", "gpt-4o-mini")

        assert result["text"] == "Hello"
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 5
        assert isinstance(result["latency_ms"], int)

    @pytest.mark.asyncio
    async def test_name(self):
        provider = self._make_provider()
        assert provider.name == "openai"

    @pytest.mark.asyncio
    async def test_api_error_raises_provider_error(self):
        provider = self._make_provider()
        provider._client = MagicMock()
        provider._client.chat.completions = MagicMock()
        provider._client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("API down")
        )

        with pytest.raises(ProviderError, match="OpenAI API call failed"):
            await provider.complete("system", "user", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------

class TestAnthropicProvider:
    """Tests for AnthropicProvider with mocked API."""

    def _make_provider(self):
        from src.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(_settings())

    @pytest.mark.asyncio
    async def test_complete_returns_correct_schema(self):
        provider = self._make_provider()
        mock_block = SimpleNamespace(text="Anthropic response")
        mock_usage = SimpleNamespace(input_tokens=12, output_tokens=8)
        mock_response = SimpleNamespace(content=[mock_block], usage=mock_usage)

        provider._client = MagicMock()
        provider._client.messages = MagicMock()
        provider._client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.complete("system", "user", "claude-haiku-4-5")

        assert result["text"] == "Anthropic response"
        assert result["input_tokens"] == 12
        assert result["output_tokens"] == 8
        assert isinstance(result["latency_ms"], int)

    @pytest.mark.asyncio
    async def test_name(self):
        provider = self._make_provider()
        assert provider.name == "anthropic"

    @pytest.mark.asyncio
    async def test_api_error_raises_provider_error(self):
        provider = self._make_provider()
        provider._client = MagicMock()
        provider._client.messages = MagicMock()
        provider._client.messages.create = AsyncMock(
            side_effect=RuntimeError("Rate limited")
        )

        with pytest.raises(ProviderError, match="Anthropic API call failed"):
            await provider.complete("system", "user", "claude-haiku-4-5")


# ---------------------------------------------------------------------------
# GeminiProvider
# ---------------------------------------------------------------------------

class TestGeminiProvider:
    """Tests for GeminiProvider with mocked API."""

    def _make_provider(self):
        with patch("src.providers.gemini_provider.genai"):
            from src.providers.gemini_provider import GeminiProvider

            return GeminiProvider(_settings())

    @pytest.mark.asyncio
    async def test_complete_returns_correct_schema(self):
        mock_usage = SimpleNamespace(prompt_token_count=15, candidates_token_count=7)
        mock_response = SimpleNamespace(text="Gemini response", usage_metadata=mock_usage)

        with patch("src.providers.gemini_provider.genai") as mock_genai:
            mock_client = mock_genai.Client.return_value
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

            from src.providers.gemini_provider import GeminiProvider

            provider = GeminiProvider(_settings())
            result = await provider.complete("system", "user", "gemini-3.5-flash-lite")

        assert result["text"] == "Gemini response"
        assert result["input_tokens"] == 15
        assert result["output_tokens"] == 7
        assert isinstance(result["latency_ms"], int)

    @pytest.mark.asyncio
    async def test_name(self):
        provider = self._make_provider()
        assert provider.name == "gemini"

    @pytest.mark.asyncio
    async def test_api_error_raises_provider_error(self):
        with patch("src.providers.gemini_provider.genai") as mock_genai:
            mock_client = mock_genai.Client.return_value
            mock_client.aio.models.generate_content = AsyncMock(
                side_effect=RuntimeError("Quota exceeded")
            )

            from src.providers.gemini_provider import GeminiProvider

            provider = GeminiProvider(_settings())

            with pytest.raises(ProviderError, match="Gemini API call failed"):
                await provider.complete("system", "user", "gemini-3.5-flash-lite")

    def test_rate_limited_detects_sdk_429(self):
        from google.genai import errors

        from src.providers.gemini_provider import _is_rate_limited

        exc = errors.ClientError(
            429,
            {
                "error": {
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "Quota exceeded",
                    "details": [{"retryInfo": {"retryDelay": "65s"}}],
                }
            },
        )
        assert _is_rate_limited(exc)

    def test_rate_limited_false_for_other_errors(self):
        from google.genai import errors

        from src.providers.gemini_provider import _is_rate_limited

        exc = errors.ClientError(
            400,
            {"error": {"status": "INVALID_ARGUMENT", "message": "Bad request"}},
        )
        assert not _is_rate_limited(exc)

    def test_retry_delay_parses_sdk_message(self):
        from google.genai import errors

        from src.providers.gemini_provider import _retry_delay_seconds

        exc = errors.ClientError(
            429,
            {
                "error": {
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "Quota exceeded",
                    "details": [{"retryInfo": {"retryDelay": "65s"}}],
                }
            },
        )
        assert _retry_delay_seconds(exc) == 66.0

    def test_retry_delay_parses_real_message(self):
        from google.genai import errors

        from src.providers.gemini_provider import _retry_delay_seconds

        exc = errors.ClientError(
            429,
            {
                "error": {
                    "status": "RESOURCE_EXHAUSTED",
                    "message": (
                        "You exceeded your current quota. "
                        "Please retry in 44.62191032s."
                    ),
                }
            },
        )
        assert _retry_delay_seconds(exc) == pytest.approx(45.62, abs=0.01)

    def test_retry_delay_falls_back(self):
        from google.genai import errors

        from src.providers.gemini_provider import _FALLBACK_RETRY_SECONDS, _retry_delay_seconds

        exc = errors.ClientError(
            429,
            {"error": {"status": "RESOURCE_EXHAUSTED", "message": "No delay hint"}},
        )
        assert _retry_delay_seconds(exc) == _FALLBACK_RETRY_SECONDS

    def test_retryable_detects_sdk_503(self):
        from google.genai import errors

        from src.providers.gemini_provider import _is_retryable

        exc = errors.ServerError(
            503,
            {
                "error": {
                    "status": "UNAVAILABLE",
                    "message": (
                        "This model is currently experiencing high demand. "
                        "Spikes in demand are usually temporary. Please try again later."
                    ),
                }
            },
        )
        assert _is_retryable(exc)

    def test_retryable_detects_sdk_500(self):
        from google.genai import errors

        from src.providers.gemini_provider import _is_retryable

        exc = errors.ServerError(
            500,
            {"error": {"status": "INTERNAL", "message": "Internal error"}},
        )
        assert _is_retryable(exc)

    def test_retryable_false_for_400(self):
        from google.genai import errors

        from src.providers.gemini_provider import _is_retryable

        exc = errors.ClientError(
            400,
            {"error": {"status": "INVALID_ARGUMENT", "message": "Bad request"}},
        )
        assert not _is_retryable(exc)

    def test_retry_backoff_grows_and_caps(self):
        from src.providers.gemini_provider import _retry_backoff_seconds

        first = _retry_backoff_seconds(1)
        fourth = _retry_backoff_seconds(4)
        late = _retry_backoff_seconds(8)

        assert first < fourth
        assert 5.0 <= first < 6.0
        assert 60.0 <= late < 61.0

    @pytest.mark.asyncio
    async def test_complete_retries_on_503_then_succeeds(self):
        from google.genai import errors

        mock_usage = SimpleNamespace(prompt_token_count=15, candidates_token_count=7)
        mock_response = SimpleNamespace(text="Gemini response", usage_metadata=mock_usage)
        server_error = errors.ServerError(
            503,
            {
                "error": {
                    "status": "UNAVAILABLE",
                    "message": "This model is currently experiencing high demand.",
                }
            },
        )

        with patch("src.providers.gemini_provider.genai") as mock_genai, patch(
            "src.providers.gemini_provider.asyncio.sleep", new=AsyncMock()
        ) as mock_sleep:
            mock_client = mock_genai.Client.return_value
            mock_client.aio.models.generate_content = AsyncMock(
                side_effect=[server_error, mock_response]
            )

            from src.providers.gemini_provider import GeminiProvider

            provider = GeminiProvider(_settings())
            result = await provider.complete("system", "user", "gemini-3.5-flash-lite")

        assert result["text"] == "Gemini response"
        assert result["input_tokens"] == 15
        mock_sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rate_limiter_enforces_cap(self):
        from src.providers.gemini_provider import _RateLimiter

        limiter = _RateLimiter(max_calls=2, window_seconds=10.0)
        await limiter.acquire()
        await limiter.acquire()

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(limiter.acquire(), timeout=0.2)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    """Tests for get_provider factory function."""

    def test_openai(self):
        from src.providers.openai_provider import OpenAIProvider

        provider = get_provider("openai", _settings())
        assert isinstance(provider, OpenAIProvider)

    def test_anthropic(self):
        from src.providers.anthropic_provider import AnthropicProvider

        provider = get_provider("anthropic", _settings())
        assert isinstance(provider, AnthropicProvider)

    def test_gemini(self):
        with patch("src.providers.gemini_provider.genai"):
            from src.providers.gemini_provider import GeminiProvider

            provider = get_provider("gemini", _settings())
            assert isinstance(provider, GeminiProvider)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("ollama", _settings())


# ---------------------------------------------------------------------------
# calculate_cost_aud
# ---------------------------------------------------------------------------

class TestCalculateCostAud:
    """Tests for the cost calculation function."""

    def test_openai_gpt4o_mini(self):
        cost = calculate_cost_aud("openai", "gpt-4o-mini", 1000, 500)
        expected = (1000 / 1000) * 0.000228 + (500 / 1000) * 0.000912
        assert abs(cost - expected) < 1e-6

    def test_anthropic_haiku(self):
        cost = calculate_cost_aud("anthropic", "claude-haiku-4-5", 2000, 1000)
        expected = (2000 / 1000) * 0.000380 + (1000 / 1000) * 0.001900
        assert abs(cost - expected) < 1e-6

    def test_gemini_flash(self):
        cost = calculate_cost_aud("gemini", "gemini-2.0-flash", 1500, 800)
        expected = (1500 / 1000) * 0.000114 + (800 / 1000) * 0.000570
        assert abs(cost - expected) < 1e-6

    def test_unknown_provider_returns_zero(self):
        assert calculate_cost_aud("unknown", "model", 100, 100) == 0.0

    def test_unknown_model_returns_zero(self):
        assert calculate_cost_aud("openai", "gpt-999", 100, 100) == 0.0


# ---------------------------------------------------------------------------
# classify_email (mocked provider)
# ---------------------------------------------------------------------------

class TestClassifyEmail:
    """Tests for classify_email with a fully mocked provider."""

    @pytest.mark.asyncio
    async def test_valid_response(self):
        settings = _settings()
        settings.model_under_test = "gpt-4o-mini"

        mock_provider = MagicMock(spec=BaseProvider)
        mock_provider.name = "openai"
        mock_provider.complete = AsyncMock(return_value={
            "text": json.dumps({"category": "billing", "summary": "Double charged."}),
            "input_tokens": 50,
            "output_tokens": 15,
            "latency_ms": 600,
        })

        prompt = PromptConfig(
            version="1.0.0",
            created_at="2025-01-01T00:00:00Z",
            system_prompt="You are a classifier.",
        )

        result = await classify_email("I was charged twice.", prompt, mock_provider, settings)

        assert result.category == "billing"
        assert result.summary == "Double charged."
        assert result.provider == "openai"
        assert result.estimated_cost_aud > 0
        assert result.latency_ms == 600

    @pytest.mark.asyncio
    async def test_invalid_json_raises(self):
        settings = _settings()
        mock_provider = MagicMock(spec=BaseProvider)
        mock_provider.name = "openai"
        mock_provider.complete = AsyncMock(return_value={
            "text": "not json",
            "input_tokens": 10,
            "output_tokens": 5,
            "latency_ms": 100,
        })

        prompt = PromptConfig(
            version="1.0.0",
            created_at="2025-01-01T00:00:00Z",
            system_prompt="prompt",
        )

        with pytest.raises(ValueError, match="invalid JSON"):
            await classify_email("test", prompt, mock_provider, settings)

    @pytest.mark.asyncio
    async def test_invalid_category_raises(self):
        settings = _settings()
        mock_provider = MagicMock(spec=BaseProvider)
        mock_provider.name = "openai"
        mock_provider.complete = AsyncMock(return_value={
            "text": json.dumps({"category": "shipping", "summary": "Something."}),
            "input_tokens": 10,
            "output_tokens": 5,
            "latency_ms": 100,
        })

        prompt = PromptConfig(
            version="1.0.0",
            created_at="2025-01-01T00:00:00Z",
            system_prompt="prompt",
        )

        with pytest.raises(ValueError, match="Invalid category"):
            await classify_email("test", prompt, mock_provider, settings)
