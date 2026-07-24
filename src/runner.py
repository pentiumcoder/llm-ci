"""Async eval runner — processes all golden dataset cases through the LLM feature."""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.config import Settings
from src.cost import calculate_cost_aud, summarise_run_cost
from src.dataset import load_dataset
from src.feature import classify_email, load_prompt
from src.models import CaseResult, EvalRun
from src.providers.factory import get_provider

logger = logging.getLogger(__name__)

_MAX_CONCURRENT = 5


async def _run_one_case(
    case,
    prompt_config,
    provider,
    settings,
    semaphore: asyncio.Semaphore,
) -> CaseResult:
    """Run classify_email for a single case, wrapped in a semaphore."""
    async with semaphore:
        result = await classify_email(
            email_text=case.input,
            prompt_config=prompt_config,
            provider=provider,
            settings=settings,
        )

        cost = calculate_cost_aud(
            provider=provider.name,
            model=settings.model_under_test,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

        return CaseResult(
            case_id=case.id,
            prompt_version=prompt_config.version,
            model=settings.model_under_test,
            provider=provider.name,
            predicted_category=result.category,
            expected_category=case.expected_category,
            category_match=(result.category == case.expected_category),
            summary_score_judge=0.0,
            summary_score_embedding=0.0,
            summary_score_keyword=0.0,
            composite_summary_score=0.0,
            judge_reason="",
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            estimated_cost_aud=cost,
            run_id="",
            timestamp=datetime.now(timezone.utc),
        )


async def run_eval(prompt_path: str, settings: Settings) -> EvalRun:
    """Run full eval: load prompt + dataset, classify all cases, persist results."""
    prompt_config = load_prompt(prompt_path)
    dataset = load_dataset(settings.golden_dataset_path)
    provider = get_provider(settings.provider, settings)

    run_id = str(uuid.uuid4())
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

    tasks = [
        _run_one_case(case, prompt_config, provider, settings, semaphore)
        for case in dataset.cases
    ]

    case_results: list[CaseResult] = []
    if tasks:
        case_results = await asyncio.gather(*tasks)

    for cr in case_results:
        cr.run_id = run_id

    total_cases = len(case_results)
    correct = sum(1 for cr in case_results if cr.category_match)
    accuracy = correct / total_cases if total_cases > 0 else 0.0

    composite_scores = [cr.composite_summary_score for cr in case_results]
    avg_composite = statistics.mean(composite_scores) if composite_scores else 0.0

    latencies = sorted([cr.latency_ms for cr in case_results]) if case_results else [0]
    p95_idx = max(0, int(len(latencies) * 0.95) - 1)
    p95_latency = latencies[p95_idx]

    total_input = sum(cr.input_tokens for cr in case_results)
    total_output = sum(cr.output_tokens for cr in case_results)
    total_cost = summarise_run_cost(case_results)

    run = EvalRun(
        run_id=run_id,
        prompt_version=prompt_config.version,
        model=settings.model_under_test,
        provider=provider.name,
        timestamp=datetime.now(timezone.utc),
        total_cases=total_cases,
        accuracy=accuracy,
        avg_composite_summary_score=avg_composite,
        p95_latency_ms=p95_latency,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_cost_aud=total_cost,
        status="PASS",
        difficulty_breakdown={},
        case_results=list(case_results),
    )

    from src.storage import init_db, save_run

    init_db(settings.db_path)
    save_run(run, settings.db_path)
    logger.info("Run %s saved to %s (%d cases, %.2f%% accuracy, AUD $%.4f)",
                run_id, settings.db_path, total_cases, accuracy * 100, total_cost)

    return run


def _print_summary(run: EvalRun) -> None:
    """Print a rich summary table of the eval run."""
    console = Console()
    table = Table(title=f"Eval Run {run.run_id[:8]}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Provider", run.provider)
    table.add_row("Model", run.model)
    table.add_row("Prompt version", run.prompt_version)
    table.add_row("Total cases", str(run.total_cases))
    table.add_row("Accuracy", f"{run.accuracy * 100:.1f}%")
    table.add_row("Avg composite score", f"{run.avg_composite_summary_score:.3f}")
    table.add_row("P95 latency", f"{run.p95_latency_ms}ms")
    table.add_row("Input tokens", f"{run.total_input_tokens:,}")
    table.add_row("Output tokens", f"{run.total_output_tokens:,}")
    table.add_row("Total cost (AUD)", f"${run.total_cost_aud:.4f}")
    table.add_row("Status", run.status)

    console.print(table)


def main() -> None:
    """CLI entry point for the eval runner."""
    parser = argparse.ArgumentParser(description="LLM-CI eval runner")
    parser.add_argument("--prompt", required=True, help="Path to prompt YAML file")
    parser.add_argument("--output", help="Output directory (unused, kept for CLI compat)")
    args = parser.parse_args()

    settings = Settings()
    run = asyncio.run(run_eval(args.prompt, settings))
    _print_summary(run)


if __name__ == "__main__":
    main()
