"""Runnable script to test classify_email with a live provider.

Requires: prompts/v1.0.0.yaml (created in Sprint 3).
Run: python scripts/test_feature.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from src.config import Settings
from src.feature import classify_email, load_prompt
from src.providers.factory import get_provider


TEST_EMAILS = [
    (
        "billing",
        "Hi, I just noticed I was charged $49.99 twice on my credit card for the same subscription. "
        "Please refund the duplicate charge ASAP.",
    ),
    (
        "technical",
        "The app keeps crashing every time I try to upload a file larger than 5MB. "
        "I'm on iOS 18.1 and the latest app version. Please fix this.",
    ),
    (
        "general",
        "Would it be possible to add dark mode to the dashboard? "
        "I think a lot of users would appreciate it. Thanks!",
    ),
]


async def main() -> None:
    """Run classify_email on test emails and print results."""
    console = Console()
    settings = Settings()
    provider = get_provider(settings.provider, settings)

    prompt_path = str(Path(__file__).resolve().parent.parent / "prompts" / "v1.0.0.yaml")
    prompt_config = load_prompt(prompt_path)

    all_passed = True
    for expected_category, email in TEST_EMAILS:
        console.rule(f"Testing: {expected_category}")
        result = await classify_email(email, prompt_config, provider, settings)
        console.print(f"  Category:   {result.category}")
        console.print(f"  Summary:    {result.summary}")
        console.print(f"  Cost AUD:   ${result.estimated_cost_aud:.6f}")
        console.print(f"  Latency:    {result.latency_ms}ms")
        console.print(f"  Provider:   {result.provider}")

        if result.category != expected_category:
            console.print(f"  [bold red]FAIL: expected {expected_category}, got {result.category}[/]")
            all_passed = False
        else:
            console.print("  [bold green]PASS[/]")

        if result.estimated_cost_aud <= 0:
            console.print("  [bold red]FAIL: cost was zero[/]")
            all_passed = False

    console.rule("Summary")
    if all_passed:
        console.print("[bold green]All 3 tests passed.[/]")
    else:
        console.print("[bold red]Some tests failed.[/]")


if __name__ == "__main__":
    asyncio.run(main())
