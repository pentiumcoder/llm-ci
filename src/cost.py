"""Per-provider pricing constants and cost calculation for LLM-CI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models import CaseResult

logger = logging.getLogger(__name__)

PRICING_AUD: dict[str, dict[str, dict[str, float]]] = {
    "openai": {
        "gpt-4o-mini": {"input_per_1k": 0.000228, "output_per_1k": 0.000912},
        "gpt-4o": {"input_per_1k": 0.00684, "output_per_1k": 0.02736},
    },
    "anthropic": {
        "claude-haiku-4-5": {"input_per_1k": 0.000380, "output_per_1k": 0.001900},
    },
    "gemini": {
        "gemini-2.0-flash": {"input_per_1k": 0.000114, "output_per_1k": 0.000570},
    },
}


def calculate_cost_aud(
    provider: str, model: str, input_tokens: int, output_tokens: int
) -> float:
    """Calculate estimated cost in AUD for a single LLM call."""
    provider_pricing = PRICING_AUD.get(provider)
    if provider_pricing is None:
        logger.warning("No pricing data for provider %s; returning 0.0", provider)
        return 0.0

    model_pricing = provider_pricing.get(model)
    if model_pricing is None:
        logger.warning("No pricing data for model %s/%s; returning 0.0", provider, model)
        return 0.0

    input_cost = (input_tokens / 1000) * model_pricing["input_per_1k"]
    output_cost = (output_tokens / 1000) * model_pricing["output_per_1k"]
    return round(input_cost + output_cost, 6)


def summarise_run_cost(case_results: list[CaseResult]) -> float:
    """Sum all estimated_cost_aud values across case results."""
    return round(sum(cr.estimated_cost_aud for cr in case_results), 6)
