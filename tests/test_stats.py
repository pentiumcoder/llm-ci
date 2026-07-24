"""Tests for src/stats.py — two-proportion z-test and significance logic."""

from __future__ import annotations

import pytest

from src.models import CaseResult, EvalRun
from src.stats import is_significant, run_ztest


def _make_case_result(case_id: str, category_match: bool) -> CaseResult:
    """Create a minimal CaseResult with the given category_match."""
    return CaseResult(
        case_id=case_id,
        prompt_version="1.0.0",
        model="gpt-4o-mini",
        provider="openai",
        predicted_category="billing",
        expected_category="billing",
        category_match=category_match,
        summary_score_judge=4.0,
        summary_score_embedding=0.8,
        summary_score_keyword=0.75,
        composite_summary_score=0.78,
        judge_reason="Accurate summary.",
        latency_ms=150,
        input_tokens=100,
        output_tokens=50,
        estimated_cost_aud=0.001,
        run_id="run-001",
        timestamp="2025-01-01T00:00:00Z",
    )


def _make_run(n: int, accuracy: float) -> EvalRun:
    """Create an EvalRun with n cases and the given accuracy (0.0-1.0)."""
    successes = int(n * accuracy)
    case_results = [
        _make_case_result(f"case_{i:03d}", i < successes)
        for i in range(n)
    ]
    return EvalRun(
        run_id="run_test",
        prompt_version="1.0.0",
        model="gpt-4o-mini",
        provider="openai",
        timestamp="2025-01-01T00:00:00Z",
        total_cases=n,
        accuracy=accuracy,
        avg_composite_summary_score=0.78,
        p95_latency_ms=150,
        total_input_tokens=n * 100,
        total_output_tokens=n * 50,
        total_cost_aud=0.001 * n,
        status="PASS",
        case_results=case_results,
    )


class TestRunZtest:
    """Tests for the run_ztest function."""

    def test_significant_drop_90_to_70(self) -> None:
        """Verify 90% to 70% drop on 100 cases is statistically significant."""
        current = _make_run(100, 0.70)
        previous = _make_run(100, 0.90)
        z_stat, p_value = run_ztest(current, previous)
        assert p_value < 0.05, f"Expected p < 0.05, got {p_value}"
        assert z_stat < 0, "Expected negative z-statistic for accuracy drop"

    def test_non_significant_1pct_drop(self) -> None:
        """Verify 1% accuracy drop on 100 cases is not significant."""
        current = _make_run(100, 0.89)
        previous = _make_run(100, 0.90)
        _z_stat, p_value = run_ztest(current, previous)
        assert p_value >= 0.05, f"Expected p >= 0.05, got {p_value}"

    def test_raises_when_current_too_few(self) -> None:
        """Verify ValueError when current run has fewer than 30 cases."""
        current = _make_run(25, 0.80)
        previous = _make_run(100, 0.90)
        with pytest.raises(ValueError, match="Current run has 25 cases"):
            run_ztest(current, previous)

    def test_raises_when_previous_too_few(self) -> None:
        """Verify ValueError when previous run has fewer than 30 cases."""
        current = _make_run(100, 0.80)
        previous = _make_run(25, 0.90)
        with pytest.raises(ValueError, match="Previous run has 25 cases"):
            run_ztest(current, previous)


class TestIsSignificant:
    """Tests for the is_significant function."""

    def test_significant_below_alpha(self) -> None:
        """Verify p=0.03 is significant at alpha=0.05."""
        assert is_significant(0.03, 0.05) is True

    def test_not_significant_above_alpha(self) -> None:
        """Verify p=0.08 is not significant at alpha=0.05."""
        assert is_significant(0.08, 0.05) is False

    def test_boundary_at_alpha(self) -> None:
        """Verify p exactly equal to alpha is not significant (strict <)."""
        assert is_significant(0.05, 0.05) is False

    def test_custom_alpha(self) -> None:
        """Verify is_significant respects custom alpha."""
        assert is_significant(0.08, 0.10) is True
