"""Difficulty-stratified regression analysis for eval runs."""

from __future__ import annotations

import logging

from src.models import CaseResult, DifficultyBreakdown, GoldenDataset

logger = logging.getLogger(__name__)


def compute_difficulty_breakdown(
    case_results: list[CaseResult], dataset: GoldenDataset
) -> DifficultyBreakdown:
    """Compute accuracy broken down by difficulty and category from case results."""
    case_map = {c.id: c for c in dataset.cases}

    stats: dict[str, dict[str, dict[str, int]]] = {}
    for cr in case_results:
        golden = case_map.get(cr.case_id)
        if golden is None:
            continue
        diff = golden.difficulty
        cat = golden.expected_category
        stats.setdefault(diff, {}).setdefault(cat, {"correct": 0, "total": 0})
        stats[diff][cat]["total"] += 1
        if cr.category_match:
            stats[diff][cat]["correct"] += 1

    breakdown: DifficultyBreakdown = {}
    for diff, categories in stats.items():
        breakdown[diff] = {}
        for cat, counts in categories.items():
            breakdown[diff][cat] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0

    logger.info("Difficulty breakdown computed: %s", breakdown)
    return breakdown


def flag_difficulty_regression(
    current: DifficultyBreakdown, previous: DifficultyBreakdown
) -> list[str]:
    """Flag if hard/edge cases regress >10% while easy cases are stable (<3% drop)."""
    warnings: list[str] = []

    easy_curr = _avg_accuracy(current.get("easy", {}))
    easy_prev = _avg_accuracy(previous.get("easy", {}))
    easy_drop = easy_prev - easy_curr

    for diff in ("hard", "edge"):
        curr_acc = _avg_accuracy(current.get(diff, {}))
        prev_acc = _avg_accuracy(previous.get(diff, {}))
        diff_drop = prev_acc - curr_acc

        if diff_drop > 0.10 and easy_drop < 0.03:
            warnings.append(
                f"Hard/edge regression: {diff} accuracy dropped "
                f"{diff_drop * 100:.1f}% while easy accuracy is stable "
                f"({easy_drop * 100:.1f}% drop)"
            )

    return warnings


def _avg_accuracy(categories: dict[str, float]) -> float:
    """Return the average accuracy across categories for a difficulty level."""
    if not categories:
        return 0.0
    return sum(categories.values()) / len(categories)
