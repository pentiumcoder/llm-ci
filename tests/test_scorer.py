"""Tests for src/scorer.py — judge, embedding, keyword, and composite scoring."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import Settings
from src.scorer import (
    JUDGE_PROMPT,
    compute_composite,
    compute_p95_latency,
    score_embedding,
    score_judge,
    score_keywords,
)


def _mock_provider(response_text: str) -> MagicMock:
    """Create a mock provider whose complete() returns the given text."""
    provider = MagicMock()
    provider.complete = AsyncMock(
        return_value={
            "text": response_text,
            "input_tokens": 50,
            "output_tokens": 20,
            "latency_ms": 100,
        }
    )
    provider.name = "openai"
    return provider


def _settings() -> Settings:
    """Return default Settings for tests."""
    return Settings()


class TestScoreJudge:
    """Tests for the score_judge async function."""

    @pytest.mark.asyncio
    async def test_valid_score(self) -> None:
        """Verify a valid judge response is parsed correctly."""
        provider = _mock_provider(json.dumps({"score": 5, "reason": "Perfect match"}))
        score, reason = await score_judge(
            "I was charged twice",
            "Customer was double charged",
            "Double billing issue",
            provider,
            _settings(),
        )
        assert score == 5.0
        assert reason == "Perfect match"

    @pytest.mark.asyncio
    async def test_clamps_low_score(self) -> None:
        """Verify scores below 1.0 are clamped to 1.0."""
        provider = _mock_provider(json.dumps({"score": 0, "reason": "Bad"}))
        score, _ = await score_judge(
            "email", "predicted", "expected", provider, _settings(),
        )
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_clamps_high_score(self) -> None:
        """Verify scores above 5.0 are clamped to 5.0."""
        provider = _mock_provider(json.dumps({"score": 7, "reason": "Too high"}))
        score, _ = await score_judge(
            "email", "predicted", "expected", provider, _settings(),
        )
        assert score == 5.0

    @pytest.mark.asyncio
    async def test_invalid_json_returns_default(self) -> None:
        """Verify invalid JSON from judge returns default score 3.0."""
        provider = _mock_provider("not json at all")
        score, reason = await score_judge(
            "email", "predicted", "expected", provider, _settings(),
        )
        assert score == 3.0
        assert "failed" in reason.lower()

    @pytest.mark.asyncio
    async def test_provider_exception_returns_default(self) -> None:
        """Verify provider exception returns default score 3.0."""
        provider = MagicMock()
        provider.complete = AsyncMock(side_effect=RuntimeError("API down"))
        score, reason = await score_judge(
            "email", "predicted", "expected", provider, _settings(),
        )
        assert score == 3.0
        assert "failed" in reason.lower()

    @pytest.mark.asyncio
    async def test_uses_judge_model_from_settings(self) -> None:
        """Verify the judge call uses settings.judge_model."""
        settings = _settings()
        settings.judge_model = "gpt-4o"
        provider = _mock_provider(json.dumps({"score": 4, "reason": "Good"}))
        await score_judge(
            "email", "predicted", "expected", provider, settings,
        )
        provider.complete.assert_called_once()
        call_kwargs = provider.complete.call_args
        assert call_kwargs.kwargs.get("model") == "gpt-4o" or call_kwargs[1].get("model") == "gpt-4o"

    def test_judge_prompt_format(self) -> None:
        """Verify JUDGE_PROMPT contains expected placeholders."""
        assert "{input}" in JUDGE_PROMPT
        assert "{predicted}" in JUDGE_PROMPT
        assert "{expected}" in JUDGE_PROMPT


class TestScoreKeywords:
    """Tests for the score_keywords function."""

    def test_all_keywords_match(self) -> None:
        """Verify score is 1.0 when all keywords are present."""
        assert score_keywords("refund charged twice", ["refund", "charged", "twice"]) == 1.0

    def test_no_keywords_match(self) -> None:
        """Verify score is 0.0 when no keywords are present."""
        assert score_keywords("login problem", ["refund", "charged"]) == 0.0

    def test_partial_match(self) -> None:
        """Verify score reflects partial keyword matches."""
        result = score_keywords("refund issue", ["refund", "charged", "twice"])
        assert abs(result - 1 / 3) < 1e-6

    def test_case_insensitive(self) -> None:
        """Verify matching is case-insensitive."""
        assert score_keywords("REFUND Charged", ["refund", "charged"]) == 1.0

    def test_empty_keywords(self) -> None:
        """Verify score is 0.0 for empty keyword list."""
        assert score_keywords("any summary", []) == 0.0

    def test_empty_summary(self) -> None:
        """Verify score is 0.0 when summary is empty."""
        assert score_keywords("", ["refund", "charged"]) == 0.0

    def test_keyword_as_substring(self) -> None:
        """Verify keyword matches as substring within words."""
        assert score_keywords("overcharged today", ["charge"]) == 1.0


class TestScoreEmbedding:
    """Tests for the score_embedding function."""

    def test_returns_float_in_range(self) -> None:
        """Verify embedding score is between 0.0 and 1.0."""
        result = score_embedding("Customer needs a refund", ["refund", "money"])
        assert 0.0 <= result <= 1.0

    def test_empty_keywords_returns_zero(self) -> None:
        """Verify score is 0.0 for empty keyword list."""
        assert score_embedding("any summary", []) == 0.0

    def test_identical_text_high_score(self) -> None:
        """Verify identical input yields a high embedding score."""
        result = score_embedding("refund", ["refund"])
        assert result >= 0.5

    def test_unrelated_text_lower_score(self) -> None:
        """Verify unrelated text yields a lower score than related text."""
        related = score_embedding("I need a refund for my account", ["refund", "account"])
        unrelated = score_embedding("The weather is nice today", ["refund", "account"])
        assert related >= unrelated


class TestComputeComposite:
    """Tests for the compute_composite function."""

    def test_perfect_scores(self) -> None:
        """Verify composite is 1.0 when all inputs are perfect."""
        result = compute_composite(5.0, 1.0, 1.0)
        assert abs(result - 1.0) < 1e-6

    def test_zero_scores(self) -> None:
        """Verify composite is 0.0 when all inputs are zero."""
        result = compute_composite(0.0, 0.0, 0.0)
        assert result == 0.0

    def test_weighted_average(self) -> None:
        """Verify composite follows the formula 0.5×(judge/5) + 0.3×embedding + 0.2×keyword."""
        result = compute_composite(3.0, 0.6, 0.8)
        expected = (3.0 / 5.0) * 0.5 + 0.6 * 0.3 + 0.8 * 0.2
        assert abs(result - expected) < 1e-6

    def test_only_judge(self) -> None:
        """Verify composite when only judge score is non-zero."""
        result = compute_composite(4.0, 0.0, 0.0)
        assert abs(result - 0.4) < 1e-6

    def test_only_embedding(self) -> None:
        """Verify composite when only embedding score is non-zero."""
        result = compute_composite(0.0, 1.0, 0.0)
        assert abs(result - 0.3) < 1e-6

    def test_only_keyword(self) -> None:
        """Verify composite when only keyword score is non-zero."""
        result = compute_composite(0.0, 0.0, 1.0)
        assert abs(result - 0.2) < 1e-6


class TestComputeP95Latency:
    """Tests for the compute_p95_latency function."""

    def test_empty_list(self) -> None:
        """Verify P95 is 0 for empty latency list."""
        assert compute_p95_latency([]) == 0

    def test_single_value(self) -> None:
        """Verify P95 returns the single value for a list of one."""
        assert compute_p95_latency([100]) == 100

    def test_known_p95(self) -> None:
        """Verify P95 is correct for a known list of latencies."""
        latencies = list(range(1, 101))  # 1..100
        result = compute_p95_latency(latencies)
        assert result == 95

    def test_sorted_input(self) -> None:
        """Verify P95 works with pre-sorted input."""
        result = compute_p95_latency([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        assert result == 90

    def test_unsorted_input(self) -> None:
        """Verify P95 works with unsorted input."""
        result = compute_p95_latency([100, 10, 50, 90, 30, 70, 20, 80, 40, 60])
        assert result == 90

    def test_all_same_values(self) -> None:
        """Verify P95 returns the value when all latencies are equal."""
        assert compute_p95_latency([150, 150, 150, 150, 150]) == 150

    def test_two_values(self) -> None:
        """Verify P95 for a minimal list of two."""
        result = compute_p95_latency([10, 20])
        assert result == 10
