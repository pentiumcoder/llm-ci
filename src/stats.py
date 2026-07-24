"""Two-proportion z-test and significance logic for regression detection."""

from __future__ import annotations

import logging

from scipy.stats import norm

from src.config import Settings
from src.models import EvalRun

logger = logging.getLogger(__name__)


def run_ztest(current: EvalRun, previous: EvalRun) -> tuple[float, float]:
    """Run two-proportion z-test comparing category_match pass rates; returns (z_statistic, p_value)."""
    if current.total_cases < Settings().min_cases_for_ztest:
        raise ValueError(
            f"Current run has {current.total_cases} cases, "
            f"minimum {Settings().min_cases_for_ztest} required for z-test"
        )
    if previous.total_cases < Settings().min_cases_for_ztest:
        raise ValueError(
            f"Previous run has {previous.total_cases} cases, "
            f"minimum {Settings().min_cases_for_ztest} required for z-test"
        )

    current_successes = sum(1 for c in current.case_results if c.category_match)
    previous_successes = sum(1 for c in previous.case_results if c.category_match)

    current_n = current.total_cases
    previous_n = previous.total_cases

    p_current = current_successes / current_n if current_n > 0 else 0.0
    p_previous = previous_successes / previous_n if previous_n > 0 else 0.0

    p_pooled = (current_successes + previous_successes) / (current_n + previous_n) if (current_n + previous_n) > 0 else 0.0

    se = (p_pooled * (1 - p_pooled) * (1 / current_n + 1 / previous_n)) ** 0.5 if (
        p_pooled * (1 - p_pooled) * (1 / current_n + 1 / previous_n) > 0
    ) else 1e-10

    z_stat = (p_current - p_previous) / se

    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    logger.info("Z-test: z=%.3f, p=%.4f (current=%.2f%%, previous=%.2f%%)",
                z_stat, p_value, p_current * 100, p_previous * 100)

    return z_stat, p_value


def is_significant(p_value: float, alpha: float = 0.05) -> bool:
    """Return True if p_value < alpha, indicating statistical significance."""
    return p_value < alpha
