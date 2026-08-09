"""Demo script — runs baseline vs candidate eval to showcase the full LLM-CI pipeline."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from src.models import EvalRun, RunDiff

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_BASELINE_PROMPT = str(_PROMPTS_DIR / "v1.0.0.yaml")
_CANDIDATE_PROMPT = str(_PROMPTS_DIR / "v1.1.0.yaml")


def _make_synthetic_baseline() -> EvalRun:
    """Create a synthetic baseline EvalRun representing v1.0.0 results."""
    return EvalRun(
        run_id="demo-baseline-00000000-0000-0000-0000-000000000000",
        prompt_version="1.0.0",
        model="gpt-4o-mini",
        provider="openai",
        timestamp=datetime(2026, 7, 24, 10, 0, 0, tzinfo=timezone.utc),
        total_cases=100,
        accuracy=0.87,
        avg_composite_summary_score=0.82,
        p95_latency_ms=1200,
        total_input_tokens=12500,
        total_output_tokens=4800,
        total_cost_aud=0.0145,
        status="PASS",
        difficulty_breakdown={
            "easy": {"billing": 1.0, "technical": 0.95, "account": 1.0, "general": 0.9},
            "medium": {"billing": 0.9, "technical": 0.85, "account": 0.9, "general": 0.8},
            "hard": {"billing": 0.8, "technical": 0.7, "account": 0.75, "general": 0.7},
            "edge": {"billing": 0.7, "technical": 0.6, "account": 0.65, "general": 0.6},
        },
        case_results=[],
    )


def _make_synthetic_candidate() -> EvalRun:
    """Create a synthetic candidate EvalRun representing v1.1.0 results with regression."""
    return EvalRun(
        run_id="demo-candidate-00000000-0000-0000-0000-000000000000",
        prompt_version="1.1.0",
        model="gpt-4o-mini",
        provider="openai",
        timestamp=datetime(2026, 7, 25, 10, 0, 0, tzinfo=timezone.utc),
        total_cases=100,
        accuracy=0.79,
        avg_composite_summary_score=0.74,
        p95_latency_ms=1500,
        total_input_tokens=13200,
        total_output_tokens=5100,
        total_cost_aud=0.0178,
        status="WARN",
        difficulty_breakdown={
            "easy": {"billing": 1.0, "technical": 0.95, "account": 1.0, "general": 0.9},
            "medium": {"billing": 0.8, "technical": 0.75, "account": 0.8, "general": 0.7},
            "hard": {"billing": 0.6, "technical": 0.5, "account": 0.55, "general": 0.5},
            "edge": {"billing": 0.5, "technical": 0.4, "account": 0.45, "general": 0.4},
        },
        case_results=[],
    )


def _run_dry() -> tuple[EvalRun, EvalRun, RunDiff]:
    """Run comparison using synthetic data without API calls."""
    from src.comparator import compute_diff
    from src.config import Settings
    from src.dataset import load_dataset

    baseline = _make_synthetic_baseline()
    candidate = _make_synthetic_candidate()

    settings = Settings()
    dataset = load_dataset(settings.golden_dataset_path)
    diff = compute_diff(candidate, baseline, settings, dataset)
    return baseline, candidate, diff


async def _run_live() -> tuple[EvalRun, EvalRun, RunDiff]:
    """Run baseline and candidate evals through the full pipeline."""
    from src.comparator import compute_diff
    from src.config import Settings
    from src.dataset import load_dataset
    from src.runner import run_eval

    settings = Settings()
    console = Console()

    with console.status("[bold blue]Running baseline eval (v1.0.0)..."):
        baseline = await run_eval(_BASELINE_PROMPT, settings)

    with console.status("[bold blue]Running candidate eval (v1.1.0)..."):
        candidate = await run_eval(_CANDIDATE_PROMPT, settings)

    dataset = load_dataset(settings.golden_dataset_path)
    diff = compute_diff(candidate, baseline, settings, dataset)
    return baseline, candidate, diff


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the demo runner."""
    parser = argparse.ArgumentParser(
        description="LLM-CI demo: baseline vs candidate comparison",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic data instead of calling LLM APIs",
    )
    return parser.parse_args()


_STATUS_STYLES = {"PASS": "bold green", "WARN": "bold yellow", "FAIL": "bold red"}


def _print_comparison(
    baseline: EvalRun,
    candidate: EvalRun,
    diff: RunDiff,
    console: Console,
) -> None:
    """Print a Rich summary of the baseline vs candidate comparison."""
    table = Table(title="Demo Comparison", show_lines=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value")

    status_style = _STATUS_STYLES.get(diff.status, "white")
    table.add_row("Status", f"[{status_style}]{diff.status}[/{status_style}]")

    if diff.accuracy_delta is not None:
        pct = diff.accuracy_delta * 100
        delta_style = "red" if pct < 0 else "green"
        table.add_row(
            "Accuracy Delta",
            f"[{delta_style}]{pct:+.1f}%[/{delta_style}]"
            f"  ({baseline.accuracy * 100:.1f}% -> {candidate.accuracy * 100:.1f}%)",
        )

    if diff.cost_delta_aud is not None:
        delta_style = "red" if diff.cost_delta_aud > 0 else "green"
        table.add_row(
            "Cost Delta (AUD)",
            f"[{delta_style}]${diff.cost_delta_aud:+.4f}[/{delta_style}]"
            f"  (${baseline.total_cost_aud:.4f} -> ${candidate.total_cost_aud:.4f})",
        )

    if diff.regression_cases:
        table.add_row("Regressions", f"{len(diff.regression_cases)} cases")

    if diff.difficulty_warnings:
        for warning in diff.difficulty_warnings:
            table.add_row("[red]Difficulty Warning[/red]", warning)

    report_path = f"results/report_{candidate.run_id[:8]}.html"
    table.add_row("Report", report_path)

    console.print()
    console.print(table)
    console.print()


def main() -> None:
    """Entry point for the demo runner."""
    args = _parse_args()
    console = Console()

    if args.dry_run:
        console.print(
            "[bold yellow]DRY RUN MODE[/bold yellow]"
            " — using synthetic data, no API calls"
        )
        baseline, candidate, diff = _run_dry()
    else:
        console.print(
            "[bold green]LIVE RUN MODE[/bold green]"
            " — calling LLM APIs"
        )
        baseline, candidate, diff = asyncio.run(_run_live())

    _print_comparison(baseline, candidate, diff, console)

    if diff.status == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
