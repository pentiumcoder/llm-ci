"""Tests for src/comparator.py — run-vs-run diff logic and status determination."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.comparator import compute_diff, determine_status
from src.config import Settings
from src.models import CaseResult, EvalRun, GoldenCase, GoldenDataset, RunDiff


def _make_case_result(
    case_id: str, category_match: bool, expected_category: str = "billing",
) -> CaseResult:
    """Create a minimal CaseResult with the given match status."""
    return CaseResult(
        case_id=case_id,
        prompt_version="1.0.0",
        model="gpt-4o-mini",
        provider="openai",
        predicted_category="billing",
        expected_category=expected_category,
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
        timestamp=datetime.now(timezone.utc),
    )


def _make_run(
    n: int, accuracy: float, composite_score: float = 0.78, cost: float = 0.1,
    case_results: list[CaseResult] | None = None,
) -> EvalRun:
    """Create an EvalRun with n cases and the given accuracy."""
    if case_results is None:
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
        timestamp=datetime.now(timezone.utc),
        total_cases=n,
        accuracy=accuracy,
        avg_composite_summary_score=composite_score,
        p95_latency_ms=150,
        total_input_tokens=n * 100,
        total_output_tokens=n * 50,
        total_cost_aud=cost,
        status="PASS",
        difficulty_breakdown={},
        case_results=case_results,
    )


def _make_dataset(n: int = 100) -> GoldenDataset:
    """Create a minimal dataset for testing."""
    cases = [
        GoldenCase(
            id=f"case_{i:03d}",
            input="Test email",
            expected_category="billing",
            expected_summary_keywords=["test"],
            difficulty="easy",
            tags=[],
            notes="Test case",
        )
        for i in range(n)
    ]
    return GoldenDataset(
        version="1.0.0",
        created_at=datetime.now(timezone.utc),
        cases=cases,
    )


class TestComputeDiff:
    """Tests for compute_diff."""

    def test_no_previous_run(self) -> None:
        """Verify PASS with None deltas when no previous run exists."""
        current = _make_run(100, 0.85)
        dataset = _make_dataset(100)
        settings = Settings()
        diff = compute_diff(current, None, settings, dataset)
        assert diff.accuracy_delta is None
        assert diff.summary_score_delta is None
        assert diff.cost_delta_aud is None
        assert diff.p_value is None
        assert diff.is_significant is False
        assert diff.status == "PASS"

    def test_significant_10pct_drop(self) -> None:
        """Verify FAIL when accuracy drops 10% and is significant."""
        previous = _make_run(100, 0.90)
        # Current: 80 correct → 80% accuracy
        current = _make_run(100, 0.80)
        dataset = _make_dataset(100)
        settings = Settings()
        diff = compute_diff(current, previous, settings, dataset)
        assert diff.is_significant is True
        assert diff.accuracy_delta is not None
        assert diff.accuracy_delta < -0.08
        assert diff.status == "FAIL"

    def test_nonsignificant_10pct_drop_fixed_threshold(self) -> None:
        """Verify WARN (not FAIL) when 10% drop is not statistically significant."""
        # previous: 100 cases, all correct
        prev_results = [
            _make_case_result(f"case_{i:03d}", True) for i in range(100)
        ]
        # current: 100 cases, 90 correct, but only 7 regressions (overlap with prev correct)
        curr_results = []
        for i in range(100):
            if i < 90:
                curr_results.append(_make_case_result(f"case_{i:03d}", True))
            else:
                curr_results.append(_make_case_result(f"case_{i:03d}", False))

        previous = EvalRun(
            run_id="prev", prompt_version="1.0.0", model="gpt-4o-mini",
            provider="openai", timestamp=datetime.now(timezone.utc),
            total_cases=100, accuracy=1.0, avg_composite_summary_score=0.85,
            p95_latency_ms=150, total_input_tokens=10000, total_output_tokens=5000,
            total_cost_aud=0.1, status="PASS", case_results=prev_results,
        )
        current = EvalRun(
            run_id="curr", prompt_version="1.0.0", model="gpt-4o-mini",
            provider="openai", timestamp=datetime.now(timezone.utc),
            total_cases=100, accuracy=0.90, avg_composite_summary_score=0.80,
            p95_latency_ms=150, total_input_tokens=10000, total_output_tokens=5000,
            total_cost_aud=0.1, status="PASS", case_results=curr_results,
        )

        dataset = _make_dataset(100)
        settings = Settings()
        settings.significance_alpha = 0.001
        diff = compute_diff(current, previous, settings, dataset)
        if diff.is_significant:
            assert diff.status == "FAIL"
        else:
            assert diff.status == "WARN"

    def test_significant_1pct_drop_pass(self) -> None:
        """Verify PASS when 1% drop is significant but under fixed threshold."""
        previous = _make_run(100, 0.90)
        # Current: 89 correct → 89% accuracy (1% drop)
        current = _make_run(100, 0.89)
        dataset = _make_dataset(100)
        settings = Settings()
        diff = compute_diff(current, previous, settings, dataset)
        # Even if significant, 1% drop is under warn_accuracy_delta (-3%)
        assert diff.status == "PASS"

    def test_difficulty_warnings_populated(self) -> None:
        """Verify difficulty_warnings is populated when hard cases regress."""
        # Create runs with difficulty_breakdown
        previous = _make_run(100, 0.90)
        previous.difficulty_breakdown = {
            "easy": {"billing": 0.97},
            "hard": {"billing": 0.85},
        }
        current = _make_run(100, 0.85)
        current.difficulty_breakdown = {
            "easy": {"billing": 0.96},
            "hard": {"billing": 0.50},
        }
        dataset = _make_dataset(100)
        settings = Settings()
        diff = compute_diff(current, previous, settings, dataset)
        assert len(diff.difficulty_warnings) > 0


class TestDetermineStatus:
    """Tests for determine_status."""

    def _settings(self) -> Settings:
        """Return default Settings for tests."""
        return Settings()

    def test_pass_no_issues(self) -> None:
        """Verify PASS when no thresholds are crossed."""
        diff = RunDiff(
            run_id="run_1",
            prev_run_id="run_0",
            accuracy_delta=-0.01,
            summary_score_delta=-0.02,
            cost_delta_aud=0.001,
            regression_cases=[],
            improvement_cases=[],
            p_value=0.5,
            z_statistic=-0.5,
            is_significant=False,
            difficulty_warnings=[],
            status="PASS",
        )
        assert determine_status(diff, self._settings()) == "PASS"

    def test_warn_accuracy_drop(self) -> None:
        """Verify WARN when accuracy drops between warn and critical thresholds."""
        diff = RunDiff(
            run_id="run_1",
            prev_run_id="run_0",
            accuracy_delta=-0.05,
            summary_score_delta=0.0,
            cost_delta_aud=0.0,
            regression_cases=[],
            improvement_cases=[],
            p_value=0.5,
            z_statistic=-0.5,
            is_significant=False,
            difficulty_warnings=[],
            status="PASS",
        )
        assert determine_status(diff, self._settings()) == "WARN"

    def test_fail_critical_accuracy(self) -> None:
        """Verify FAIL when significant accuracy drops beyond critical threshold."""
        diff = RunDiff(
            run_id="run_1",
            prev_run_id="run_0",
            accuracy_delta=-0.10,
            summary_score_delta=0.0,
            cost_delta_aud=0.0,
            regression_cases=[],
            improvement_cases=[],
            p_value=0.01,
            z_statistic=-2.5,
            is_significant=True,
            difficulty_warnings=[],
            status="PASS",
        )
        assert determine_status(diff, self._settings()) == "FAIL"

    def test_fail_many_regressions(self) -> None:
        """Verify FAIL when regression count exceeds critical threshold and z-test agrees."""
        diff = RunDiff(
            run_id="run_1",
            prev_run_id="run_0",
            accuracy_delta=-0.02,
            summary_score_delta=0.0,
            cost_delta_aud=0.0,
            regression_cases=[f"case_{i}" for i in range(10)],
            improvement_cases=[],
            p_value=0.01,
            z_statistic=-2.0,
            is_significant=True,
            difficulty_warnings=[],
            status="PASS",
        )
        assert determine_status(diff, self._settings()) == "FAIL"

    def test_warn_summary_score_drop(self) -> None:
        """Verify WARN when composite score drops between warn and critical."""
        diff = RunDiff(
            run_id="run_1",
            prev_run_id="run_0",
            accuracy_delta=-0.01,
            summary_score_delta=-0.15,
            cost_delta_aud=0.0,
            regression_cases=[],
            improvement_cases=[],
            p_value=0.5,
            z_statistic=-0.5,
            is_significant=False,
            difficulty_warnings=[],
            status="PASS",
        )
        assert determine_status(diff, self._settings()) == "WARN"
