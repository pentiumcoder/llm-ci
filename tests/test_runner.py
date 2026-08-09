"""Tests for the async eval runner."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.config import Settings
from src.models import GoldenCase, PromptConfig
from src.runner import _failed_case_result, _run_one_case


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings(**overrides) -> Settings:
    defaults = {
        "openai_api_key": "sk-test",
        "anthropic_api_key": "sk-ant-test",
        "gemini_api_key": "gemini-test",
        "provider": "gemini",
        "model_under_test": "gemini-3.5-flash-lite",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _case() -> GoldenCase:
    return GoldenCase(
        id="case_001",
        input="I was charged twice for the same order.",
        expected_category="billing",
        expected_summary_keywords=["refund", "charged"],
        difficulty="easy",
        tags=["typo"],
        notes="Double charge.",
    )


def _prompt() -> PromptConfig:
    return PromptConfig(
        version="v1.1.0",
        created_at=datetime.now(timezone.utc),
        system_prompt="Classify this email.",
        few_shot_examples=[],
    )


# ---------------------------------------------------------------------------
# Per-case failure tolerance
# ---------------------------------------------------------------------------

class TestRunOneCaseTolerance:
    """Tests that a failed classification degrades to a zero-scoring case."""

    @pytest.mark.asyncio
    async def test_classification_failure_returns_failed_result(self):
        provider = object()
        semaphore = asyncio.Semaphore(2)

        with patch(
            "src.runner.classify_email",
            AsyncMock(side_effect=RuntimeError("503 UNAVAILABLE")),
        ):
            result = await _run_one_case(_case(), _prompt(), provider, _settings(), semaphore)

        assert result.category_match is False
        assert result.predicted_category == "error"
        assert result.composite_summary_score == 0.0
        assert result.estimated_cost_aud == 0.0
        assert result.input_tokens == 0
        assert "503 UNAVAILABLE" in result.judge_reason

    def test_failed_case_result_builds_zero_scoring_row(self):
        result = _failed_case_result(_case(), _prompt(), _settings(), "boom")

        assert result.case_id == "case_001"
        assert result.category_match is False
        assert result.predicted_category == "error"
        assert result.summary_score_judge == 0.0
        assert result.summary_score_embedding == 0.0
        assert result.summary_score_keyword == 0.0
        assert result.composite_summary_score == 0.0
        assert result.estimated_cost_aud == 0.0
        assert result.provider == "gemini"
        assert result.prompt_version == "v1.1.0"
        assert isinstance(result.timestamp, datetime)
        assert result.timestamp.tzinfo == timezone.utc
