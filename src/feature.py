"""The LLM feature under test — customer support email classifier."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import ValidationError

from src.config import Settings
from src.cost import calculate_cost_aud
from src.models import ClassificationResult, PromptConfig
from src.providers.base import BaseProvider

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {"billing", "technical", "account", "general"}


def load_prompt(prompt_path: str) -> PromptConfig:
    """Read a YAML prompt file and return a validated PromptConfig."""
    path = Path(prompt_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PromptConfig(**raw)


async def classify_email(
    email_text: str,
    prompt_config: PromptConfig,
    provider: BaseProvider,
    settings: Settings,
) -> ClassificationResult:
    """Classify a customer support email using the given provider and prompt."""
    response = await provider.complete(
        system_prompt=prompt_config.system_prompt,
        user_message=email_text,
        model=settings.model_under_test,
    )

    text = response["text"]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in LLM response: %s", text)
        raise ValueError(f"LLM returned invalid JSON: {exc}") from exc

    category = parsed.get("category")
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category {category!r}; expected one of {VALID_CATEGORIES}")

    summary = parsed.get("summary", "")

    cost = calculate_cost_aud(
        provider=provider.name,
        model=settings.model_under_test,
        input_tokens=response["input_tokens"],
        output_tokens=response["output_tokens"],
    )

    return ClassificationResult(
        category=category,
        summary=summary,
        input_tokens=response["input_tokens"],
        output_tokens=response["output_tokens"],
        latency_ms=response["latency_ms"],
        estimated_cost_aud=cost,
        provider=provider.name,
    )
