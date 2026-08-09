"""Scoring engine — judge, embedding, keyword, and composite scoring for summaries."""

from __future__ import annotations

import json
import logging
import re

from sentence_transformers import SentenceTransformer, util

from src.config import Settings
from src.providers.base import BaseProvider

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an expert evaluator. Rate how well the predicted summary captures the same meaning as the expected summary.

Input email: {input}
Predicted summary: {predicted}
Expected summary: {expected}

Rate on a scale of 1-5:
1 = Completely different meaning
2 = Mostly different
3 = Partially similar
4 = Very similar
5 = Nearly identical meaning

Respond ONLY with JSON: {{"score": <int>, "reason": "<brief>"}}"""

_embed_model: SentenceTransformer | None = None


def _parse_json_response(text: str) -> dict:
    """Extract a JSON object from a model response, tolerating markdown fences."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match is None:
        raise ValueError(f"No JSON object found in response: {text!r}")
    return json.loads(match.group(0))


def _get_embed_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Return a singleton SentenceTransformer instance, loading it on first call."""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(model_name)
    return _embed_model


async def score_judge(
    email_text: str,
    predicted_summary: str,
    expected_summary: str,
    provider: BaseProvider,
    settings: Settings,
) -> tuple[float, str]:
    """Score a summary using LLM-as-judge, returning (score 1.0–5.0, reason)."""
    prompt = JUDGE_PROMPT.format(
        input=email_text,
        predicted=predicted_summary,
        expected=expected_summary,
    )
    try:
        response = await provider.complete(
            system_prompt="You are an expert evaluator.",
            user_message=prompt,
            model=settings.judge_model,
        )
        parsed = _parse_json_response(response["text"])
        score = float(parsed.get("score", 3))
        reason = parsed.get("reason", "")
        score = max(1.0, min(5.0, score))
        return score, reason
    except Exception:
        logger.exception("Judge scoring failed; defaulting to score=3.0")
        return 3.0, "Judge scoring failed"


def score_embedding(generated_summary: str, reference_keywords: list[str]) -> float:
    """Compute cosine similarity between generated summary and keyword centroid, normalised to 0–1."""
    if not reference_keywords:
        return 0.0
    model = _get_embed_model()
    reference = " ".join(reference_keywords)
    embeddings = model.encode([generated_summary, reference])
    similarity = float(util.cos_sim(embeddings[0], embeddings[1]))
    return (similarity + 1.0) / 2.0


def score_keywords(generated_summary: str, expected_keywords: list[str]) -> float:
    """Compute proportion of expected keywords present in generated summary (case-insensitive)."""
    if not expected_keywords:
        return 0.0
    summary_lower = generated_summary.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in summary_lower)
    return matches / len(expected_keywords)


def compute_composite(
    judge_score: float,
    embedding_score: float,
    keyword_score: float,
) -> float:
    """Compute weighted composite score: 0.5×(judge/5) + 0.3×embedding + 0.2×keyword."""
    return (judge_score / 5.0) * 0.5 + embedding_score * 0.3 + keyword_score * 0.2


def compute_p95_latency(latencies: list[int]) -> int:
    """Compute P95 latency from a list of latency values in milliseconds."""
    if not latencies:
        return 0
    sorted_lat = sorted(latencies)
    idx = max(0, int(len(sorted_lat) * 0.95) - 1)
    return sorted_lat[idx]
