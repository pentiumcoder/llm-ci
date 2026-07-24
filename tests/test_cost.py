"""Tests for src/cost.py — pricing constants and cost calculation."""

from datetime import datetime, timezone

from src.cost import PRICING_AUD, calculate_cost_aud, summarise_run_cost
from src.models import CaseResult


class TestCalculateCostAud:
    """Tests for the calculate_cost_aud function."""

    def test_openai_gpt4o_mini_known_tokens(self) -> None:
        """Verify correct AUD cost for openai/gpt-4o-mini with known token counts."""
        result = calculate_cost_aud("openai", "gpt-4o-mini", input_tokens=1000, output_tokens=500)
        expected = (1000 / 1000) * 0.000228 + (500 / 1000) * 0.000912
        assert result == round(expected, 6)

    def test_anthropic_claude_haiku_known_tokens(self) -> None:
        """Verify correct AUD cost for anthropic/claude-haiku-4-5 with known token counts."""
        result = calculate_cost_aud("anthropic", "claude-haiku-4-5", input_tokens=2000, output_tokens=1000)
        expected = (2000 / 1000) * 0.000380 + (1000 / 1000) * 0.001900
        assert result == round(expected, 6)

    def test_gemini_flash_known_tokens(self) -> None:
        """Verify correct AUD cost for gemini/gemini-2.0-flash with known token counts."""
        result = calculate_cost_aud("gemini", "gemini-2.0-flash", input_tokens=3000, output_tokens=2000)
        expected = (3000 / 1000) * 0.000114 + (2000 / 1000) * 0.000570
        assert result == round(expected, 6)

    def test_zero_tokens(self) -> None:
        """Verify zero cost when token counts are zero."""
        result = calculate_cost_aud("openai", "gpt-4o-mini", input_tokens=0, output_tokens=0)
        assert result == 0.0

    def test_unknown_provider_returns_zero(self) -> None:
        """Verify unknown provider returns 0.0 without raising."""
        result = calculate_cost_aud("unknown_provider", "some-model", input_tokens=1000, output_tokens=500)
        assert result == 0.0

    def test_unknown_model_returns_zero(self) -> None:
        """Verify unknown model returns 0.0 without raising."""
        result = calculate_cost_aud("openai", "nonexistent-model", input_tokens=1000, output_tokens=500)
        assert result == 0.0

    def test_pricing_dict_structure(self) -> None:
        """Verify PRICING_AUD has the expected providers and models."""
        assert "openai" in PRICING_AUD
        assert "anthropic" in PRICING_AUD
        assert "gemini" in PRICING_AUD
        assert "gpt-4o-mini" in PRICING_AUD["openai"]
        assert "gpt-4o" in PRICING_AUD["openai"]
        assert "claude-haiku-4-5" in PRICING_AUD["anthropic"]
        assert "gemini-2.0-flash" in PRICING_AUD["gemini"]


class TestSummariseRunCost:
    """Tests for the summarise_run_cost function."""

    def _make_case_result(self, cost: float) -> CaseResult:
        """Create a minimal CaseResult with the given cost."""
        return CaseResult(
            case_id="case_001",
            prompt_version="1.0.0",
            model="gpt-4o-mini",
            provider="openai",
            predicted_category="billing",
            expected_category="billing",
            category_match=True,
            summary_score_judge=4.0,
            summary_score_embedding=0.8,
            summary_score_keyword=0.75,
            composite_summary_score=0.78,
            judge_reason="Accurate summary.",
            latency_ms=150,
            input_tokens=100,
            output_tokens=50,
            estimated_cost_aud=cost,
            run_id="run-001",
            timestamp=datetime.now(timezone.utc),
        )

    def test_single_case(self) -> None:
        """Verify summarise_run_cost returns the cost of a single case."""
        case = self._make_case_result(cost=0.001234)
        assert summarise_run_cost([case]) == 0.001234

    def test_multiple_cases(self) -> None:
        """Verify summarise_run_cost sums costs across multiple cases."""
        cases = [
            self._make_case_result(cost=0.001),
            self._make_case_result(cost=0.002),
            self._make_case_result(cost=0.003),
        ]
        assert summarise_run_cost(cases) == 0.006

    def test_empty_list(self) -> None:
        """Verify summarise_run_cost returns 0.0 for an empty list."""
        assert summarise_run_cost([]) == 0.0
