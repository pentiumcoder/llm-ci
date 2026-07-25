"""Tests for src/comparator.py — run-vs-run diff logic and status determination."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.comparator import compute_diff, determine_status
from src.config import Settings
from src.drift import check_slow_drift, get_drift_summary
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


def _make_drift_run(accuracy: float, hours_ago: int) -> EvalRun:
    """Create a minimal EvalRun for drift testing with a specific timestamp."""
    return EvalRun(
        run_id=f"run_{hours_ago:03d}",
        prompt_version="1.0.0",
        model="gpt-4o-mini",
        provider="openai",
        timestamp=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
        total_cases=100,
        accuracy=accuracy,
        avg_composite_summary_score=0.78,
        p95_latency_ms=150,
        total_input_tokens=10000,
        total_output_tokens=5000,
        total_cost_aud=0.1,
        status="PASS",
        difficulty_breakdown={},
        case_results=[],
    )


class TestCheckSlowDrift:
    """Tests for check_slow_drift."""

    def test_declining_runs_detected(self) -> None:
        """Verify drift detected when 7 runs decline from 95% to 83%."""
        settings = Settings()
        settings.slow_drift_window = 7
        settings.slow_drift_threshold = -0.05
        # Newest first: 83%, 85%, 87%, 89%, 91%, 93%, 95%
        runs = [_make_drift_run(acc, i) for i, acc in enumerate([0.83, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95])]
        assert check_slow_drift(runs, settings) is True

    def test_stable_runs_no_drift(self) -> None:
        """Verify no drift when 7 runs are stable at 90%."""
        settings = Settings()
        settings.slow_drift_window = 7
        settings.slow_drift_threshold = -0.05
        runs = [_make_drift_run(0.90, i) for i in range(7)]
        assert check_slow_drift(runs, settings) is False

    def test_fewer_than_window_returns_false(self) -> None:
        """Verify False when fewer runs than window size."""
        settings = Settings()
        settings.slow_drift_window = 7
        runs = [_make_drift_run(0.83, i) for i in range(5)]
        assert check_slow_drift(runs, settings) is False


class TestGetDriftSummary:
    """Tests for get_drift_summary."""

    def test_drift_detected_message(self) -> None:
        """Verify human-readable message when drift is detected."""
        settings = Settings()
        settings.slow_drift_window = 7
        settings.slow_drift_threshold = -0.05
        runs = [_make_drift_run(acc, i) for i, acc in enumerate([0.83, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95])]
        summary = get_drift_summary(runs, settings)
        assert "Slow drift detected" in summary
        assert "95.0%" in summary
        assert "89.0%" in summary
        assert "7 runs" in summary

    def test_no_drift_message(self) -> None:
        """Verify 'No slow drift detected' when no drift."""
        settings = Settings()
        settings.slow_drift_window = 7
        settings.slow_drift_threshold = -0.05
        runs = [_make_drift_run(0.90, i) for i in range(7)]
        summary = get_drift_summary(runs, settings)
        assert summary == "No slow drift detected"
