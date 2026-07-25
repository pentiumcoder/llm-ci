"""Rolling average / slow drift detection for eval run accuracy trends."""

from __future__ import annotations

import logging
import statistics

from src.config import Settings
from src.models import EvalRun

logger = logging.getLogger(__name__)


def check_slow_drift(runs: list[EvalRun], settings: Settings) -> bool:
    """Detect if the rolling average accuracy has drifted more than the threshold from the window peak."""
    window = runs[: settings.slow_drift_window]
    if len(window) < settings.slow_drift_window:
        return False

    peak_accuracy = max(run.accuracy for run in window)
    rolling_avg = statistics.mean(run.accuracy for run in window)
    drift = peak_accuracy - rolling_avg

    if drift > abs(settings.slow_drift_threshold):
        logger.warning("Slow drift detected: peak=%.1f%%, rolling_avg=%.1f%%, drift=%.1f%%",
                       peak_accuracy * 100, rolling_avg * 100, drift * 100)
        return True
    return False


def get_drift_summary(runs: list[EvalRun], settings: Settings) -> str:
    """Return a human-readable one-liner describing the drift status."""
    if not check_slow_drift(runs, settings):
        return "No slow drift detected"

    window = runs[: settings.slow_drift_window]
    peak_accuracy = max(run.accuracy for run in window)
    rolling_avg = statistics.mean(run.accuracy for run in window)

    return (
        f"Slow drift detected: accuracy fell from {peak_accuracy * 100:.1f}% "
        f"peak to {rolling_avg * 100:.1f}% avg over last {len(window)} runs"
    )
