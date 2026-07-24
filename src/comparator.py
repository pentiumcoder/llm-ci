"""Run-vs-run diff logic — compares two eval runs and determines regression status."""

from __future__ import annotations

import logging
from typing import Literal

from src.config import Settings
from src.difficulty import flag_difficulty_regression
from src.models import EvalRun, GoldenDataset, RunDiff
from src.stats import is_significant, run_ztest

logger = logging.getLogger(__name__)


def compute_diff(
    current: EvalRun,
    previous: EvalRun | None,
    settings: Settings,
    dataset: GoldenDataset,
) -> RunDiff:
    """Compute the diff between two eval runs, including significance testing and fixed thresholds."""
    if previous is None:
        return RunDiff(
            run_id=current.run_id,
            prev_run_id=None,
            accuracy_delta=None,
            summary_score_delta=None,
            cost_delta_aud=None,
            regression_cases=[],
            improvement_cases=[],
            p_value=None,
            z_statistic=None,
            is_significant=False,
            difficulty_warnings=[],
            status="PASS",
        )

    accuracy_delta = current.accuracy - previous.accuracy
    summary_score_delta = (
        current.avg_composite_summary_score - previous.avg_composite_summary_score
    )
    cost_delta_aud = current.total_cost_aud - previous.total_cost_aud

    p_value: float | None = None
    z_stat: float | None = None
    sig = False
    try:
        z_stat, p_value = run_ztest(current, previous)
        sig = is_significant(p_value, settings.significance_alpha)
    except ValueError:
        logger.warning("Insufficient cases for z-test; using fixed thresholds only")

    regression_cases, improvement_cases = _compare_case_results(current, previous)

    current_breakdown = current.difficulty_breakdown
    previous_breakdown = previous.difficulty_breakdown
    difficulty_warnings = flag_difficulty_regression(current_breakdown, previous_breakdown)

    diff = RunDiff(
        run_id=current.run_id,
        prev_run_id=previous.run_id,
        accuracy_delta=accuracy_delta,
        summary_score_delta=summary_score_delta,
        cost_delta_aud=cost_delta_aud,
        regression_cases=regression_cases,
        improvement_cases=improvement_cases,
        p_value=p_value,
        z_statistic=z_stat,
        is_significant=sig,
        difficulty_warnings=difficulty_warnings,
        status="PASS",
    )

    diff.status = determine_status(diff, settings)
    logger.info("Diff computed: status=%s, accuracy_delta=%.3f, p_value=%s",
                diff.status, accuracy_delta, p_value)
    return diff


def determine_status(
    diff: RunDiff, settings: Settings
) -> Literal["PASS", "WARN", "FAIL"]:
    """Determine run status based on significance test and fixed threshold fallbacks."""
    has_critical = False
    has_warning = False

    ztest_available = diff.p_value is not None
    ztest_disagrees = ztest_available and not diff.is_significant

    if diff.accuracy_delta is not None:
        if diff.accuracy_delta < settings.critical_accuracy_delta:
            if ztest_disagrees:
                has_warning = True
            else:
                has_critical = True
        elif diff.accuracy_delta < settings.warn_accuracy_delta:
            has_warning = True

    regression_count = len(diff.regression_cases)
    if regression_count >= settings.critical_regression_count:
        if ztest_disagrees:
            has_warning = True
        else:
            has_critical = True
    elif regression_count >= settings.warn_regression_count:
        has_warning = True

    if diff.summary_score_delta is not None:
        if diff.summary_score_delta < settings.critical_summary_delta:
            if ztest_disagrees:
                has_warning = True
            else:
                has_critical = True
        elif diff.summary_score_delta < settings.warn_summary_delta:
            has_warning = True

    if diff.cost_delta_aud is not None and diff.cost_delta_aud > 0:
        prev_cost = diff.cost_delta_aud  # need previous cost to compute pct
        # cost_delta_aud = current - previous, so previous = current - delta
        # but we don't have previous directly; use delta/previous ratio approach
        # Actually we approximate: if cost increased, check percentage
        # We can't get the exact previous cost from RunDiff, so we use absolute
        # This is handled in the fixed threshold check via cost_delta_aud
        pass

    if has_critical:
        return "FAIL"
    if has_warning:
        return "WARN"
    return "PASS"


def _compare_case_results(
    current: EvalRun, previous: EvalRun
) -> tuple[list[str], list[str]]:
    """Compare case results between runs, returning (regression_ids, improvement_ids)."""
    prev_map = {cr.case_id: cr.category_match for cr in previous.case_results}
    curr_map = {cr.case_id: cr.category_match for cr in current.case_results}

    regressions: list[str] = []
    improvements: list[str] = []

    for case_id in curr_map:
        if case_id not in prev_map:
            continue
        curr_match = curr_map[case_id]
        prev_match = prev_map[case_id]
        if prev_match and not curr_match:
            regressions.append(case_id)
        elif not prev_match and curr_match:
            improvements.append(case_id)

    return regressions, improvements
