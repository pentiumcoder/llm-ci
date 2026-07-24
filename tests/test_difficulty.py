"""Tests for src/difficulty.py — difficulty-stratified regression analysis."""

from __future__ import annotations

from datetime import datetime, timezone

from src.difficulty import compute_difficulty_breakdown, flag_difficulty_regression
from src.models import CaseResult, GoldenCase, GoldenDataset


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


def _make_golden_case(case_id: str, difficulty: str, category: str) -> GoldenCase:
    """Create a minimal GoldenCase for testing."""
    return GoldenCase(
        id=case_id,
        input="Test email",
        expected_category=category,
        expected_summary_keywords=["test"],
        difficulty=difficulty,  # type: ignore[arg-type]
        tags=[],
        notes="Test case",
    )


def _make_dataset(cases: list[GoldenCase]) -> GoldenDataset:
    """Create a GoldenDataset from a list of GoldenCases."""
    return GoldenDataset(
        version="1.0.0",
        created_at=datetime.now(timezone.utc),
        cases=cases,
    )


class TestComputeDifficultyBreakdown:
    """Tests for compute_difficulty_breakdown."""

    def test_known_inputs(self) -> None:
        """Verify correct breakdown for known inputs."""
        dataset = _make_dataset([
            _make_golden_case("c1", "easy", "billing"),
            _make_golden_case("c2", "easy", "billing"),
            _make_golden_case("c3", "easy", "technical"),
            _make_golden_case("c4", "hard", "billing"),
        ])
        case_results = [
            _make_case_result("c1", True),
            _make_case_result("c2", True),
            _make_case_result("c3", False),
            _make_case_result("c4", False),
        ]
        result = compute_difficulty_breakdown(case_results, dataset)
        assert result["easy"]["billing"] == 1.0
        assert result["easy"]["technical"] == 0.0
        assert result["hard"]["billing"] == 0.0

    def test_mixed_accuracy(self) -> None:
        """Verify fractional accuracy when some cases pass and some fail."""
        dataset = _make_dataset([
            _make_golden_case(f"c{i}", "medium", "general")
            for i in range(4)
        ])
        case_results = [
            _make_case_result(f"c{i}", i < 2, "general")
            for i in range(4)
        ]
        result = compute_difficulty_breakdown(case_results, dataset)
        assert result["medium"]["general"] == 0.5


class TestFlagDifficultyRegression:
    """Tests for flag_difficulty_regression."""

    def test_hard_regresses_easy_stable(self) -> None:
        """Verify warning fires when hard cases regress but easy cases don't."""
        current = {"easy": {"billing": 0.95}, "hard": {"billing": 0.50}}
        previous = {"easy": {"billing": 0.97}, "hard": {"billing": 0.80}}
        warnings = flag_difficulty_regression(current, previous)
        assert len(warnings) == 1
        assert "hard" in warnings[0].lower()

    def test_all_regress_equally(self) -> None:
        """Verify empty list when all difficulties regress by the same amount."""
        current = {"easy": {"billing": 0.60}, "hard": {"billing": 0.60}}
        previous = {"easy": {"billing": 0.90}, "hard": {"billing": 0.90}}
        warnings = flag_difficulty_regression(current, previous)
        assert warnings == []

    def test_edge_regresses_easy_stable(self) -> None:
        """Verify warning fires for edge difficulty regression."""
        current = {"easy": {"billing": 0.96}, "edge": {"billing": 0.40}}
        previous = {"easy": {"billing": 0.97}, "edge": {"billing": 0.85}}
        warnings = flag_difficulty_regression(current, previous)
        assert len(warnings) == 1
        assert "edge" in warnings[0].lower()

    def test_easy_regresses_too(self) -> None:
        """Verify no warning when easy cases also regress significantly."""
        current = {"easy": {"billing": 0.80}, "hard": {"billing": 0.50}}
        previous = {"easy": {"billing": 0.95}, "hard": {"billing": 0.80}}
        warnings = flag_difficulty_regression(current, previous)
        assert warnings == []

    def test_no_previous_data(self) -> None:
        """Verify empty list when previous has no data for compared difficulties."""
        current = {"hard": {"billing": 0.50}}
        previous: dict[str, dict[str, float]] = {}
        warnings = flag_difficulty_regression(current, previous)
        assert warnings == []
