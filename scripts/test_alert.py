"""Send a synthetic test Slack alert to verify webhook configuration."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alerter import send_slack_alert
from src.config import Settings
from src.models import EvalRun, RunDiff

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _make_mock_run() -> EvalRun:
    """Create a mock EvalRun for testing the Slack alert."""
    return EvalRun(
        run_id="test-alert-00000000-0000-0000-0000-000000000000",
        prompt_version="1.0.0",
        model="gpt-4o-mini",
        provider="openai",
        timestamp=datetime.now(timezone.utc),
        total_cases=100,
        accuracy=0.87,
        avg_composite_summary_score=0.78,
        p95_latency_ms=1800,
        total_input_tokens=12500,
        total_output_tokens=4800,
        total_cost_aud=0.0183,
        status="WARN",
        difficulty_breakdown={
            "easy": {"billing": 1.0, "technical": 0.95, "account": 1.0, "general": 0.9},
            "hard": {"billing": 0.7, "technical": 0.6, "account": 0.65, "general": 0.7},
        },
        case_results=[],
    )


def _make_mock_diff() -> RunDiff:
    """Create a mock RunDiff for testing the Slack alert."""
    return RunDiff(
        run_id="test-alert-00000000-0000-0000-0000-000000000000",
        prev_run_id="prev-run-id",
        accuracy_delta=-0.04,
        summary_score_delta=-0.08,
        cost_delta_aud=0.003,
        regression_cases=["case_012", "case_034", "case_056", "case_078"],
        improvement_cases=["case_022"],
        p_value=0.032,
        z_statistic=-2.15,
        is_significant=True,
        difficulty_warnings=[
            "Hard/edge regression: hard accuracy dropped 12.5% while easy accuracy is stable (0.5% drop)"
        ],
        status="WARN",
    )


def main() -> None:
    """Send a synthetic test alert to the configured Slack webhook."""
    settings = Settings()
    if not settings.slack_webhook_url:
        logger.error("SLACK_WEBHOOK_URL not set. Add it to your .env file.")
        sys.exit(1)

    run = _make_mock_run()
    diff = _make_mock_diff()

    success = send_slack_alert(run, diff, "https://example.com/report", settings, drift_warning=True)
    if success:
        logger.info("Test alert sent successfully!")
    else:
        logger.error("Test alert failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
