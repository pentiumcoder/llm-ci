"""Slack webhook sender — posts eval run alerts to a Slack channel."""

from __future__ import annotations

import json
import logging
import urllib.request

from src.config import Settings
from src.models import EvalRun, RunDiff

logger = logging.getLogger(__name__)


def send_slack_alert(
    run: EvalRun,
    diff: RunDiff,
    report_path: str,
    settings: Settings,
    drift_warning: bool,
) -> bool:
    """Send a Slack Block Kit alert for the eval run; return True on success."""
    if not settings.slack_webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not set; skipping alert")
        return False

    prev_accuracy_pct = _prev_accuracy_pct(diff, run)
    run_cost = f"{run.total_cost_aud:.4f}"
    prev_cost = _prev_cost_str(diff, run)
    p_value_str = f"{diff.p_value:.3f}" if diff.p_value is not None else "N/A"
    significant_label = "significant ⚠️" if diff.is_significant else "not significant"
    regression_summary = format_regression_summary(diff)
    difficulty_warning = (
        "⚠️ Hard/edge cases regressing faster than easy cases"
        if diff.difficulty_warnings
        else ""
    )

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"LLM-CI Eval Run: {run.status}"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Prompt version:*\n{run.prompt_version}"},
                {"type": "mrkdwn", "text": f"*Provider / Model:*\n{run.provider} / {run.model}"},
                {"type": "mrkdwn", "text": f"*Accuracy:*\n{run.accuracy * 100:.1f}% (was {prev_accuracy_pct})"},
                {"type": "mrkdwn", "text": f"*Significance:*\np={p_value_str} ({significant_label})"},
                {"type": "mrkdwn", "text": f"*Regressions:*\n{len(diff.regression_cases)} cases"},
                {"type": "mrkdwn", "text": f"*Est. cost:*\nAUD ${run_cost} (was ${prev_cost})"},
                {"type": "mrkdwn", "text": f"*Run ID:*\n{run.run_id[:8]}"},
            ],
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": regression_summary}},
    ]
    if difficulty_warning:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": difficulty_warning}})
    if report_path.startswith(("http://", "https://")):
        blocks.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "action_id": "view_report",
                        "text": {"type": "plain_text", "text": "View Full Report"},
                        "url": report_path,
                    }
                ],
            }
        )

    payload = json.dumps({"blocks": blocks}).encode("utf-8")

    try:
        req = urllib.request.Request(
            settings.slack_webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = resp.status
        logger.info("Slack alert sent successfully (HTTP %d)", status)
        return True
    except Exception:
        logger.exception("Failed to send Slack alert")
        return False


def format_regression_summary(diff: RunDiff) -> str:
    """Return a human-readable one-line summary of regressions and improvements."""
    reg_count = len(diff.regression_cases)
    imp_count = len(diff.improvement_cases)
    parts: list[str] = []
    if reg_count > 0:
        parts.append(f"{reg_count} regression{'s' if reg_count != 1 else ''}")
    if imp_count > 0:
        parts.append(f"{imp_count} improvement{'s' if imp_count != 1 else ''}")
    if not parts:
        return "No regressions or improvements detected."
    return "; ".join(parts) + "."


def _prev_accuracy_pct(diff: RunDiff, run: EvalRun) -> str:
    """Format the previous run accuracy as a percentage string."""
    if diff.accuracy_delta is not None:
        prev = run.accuracy - diff.accuracy_delta
        return f"{prev * 100:.1f}%"
    return "N/A"


def _prev_cost_str(diff: RunDiff, run: EvalRun) -> str:
    """Format the previous run cost as a dollar string."""
    if diff.cost_delta_aud is not None:
        prev = run.total_cost_aud - diff.cost_delta_aud
        return f"{prev:.4f}"
    return "N/A"
