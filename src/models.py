"""Pydantic data models for the LLM-CI eval pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, TypeAlias

from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    """Configuration for a versioned prompt."""

    version: str
    created_at: datetime
    system_prompt: str
    few_shot_examples: list[dict[str, str]] = Field(default_factory=list)


class ClassificationResult(BaseModel):
    """Structured output from the LLM feature under test."""

    category: Literal["billing", "technical", "account", "general"]
    summary: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    estimated_cost_aud: float = 0.0
    provider: str


class GoldenCase(BaseModel):
    """A single case in the golden evaluation dataset."""

    id: str
    input: str
    expected_category: str
    expected_summary_keywords: list[str]
    difficulty: Literal["easy", "medium", "hard", "edge"]
    tags: list[str]
    notes: str
    adversarial_target: str | None = None


class GenerationMetadata(BaseModel):
    """Metadata documenting how the golden dataset was generated."""

    generator_model: str
    prompt_version: str
    temperature: float
    generation_timestamp: datetime
    total_api_calls: int
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_aud: float
    generation_mode_counts: dict[str, int]


class GoldenDataset(BaseModel):
    """The complete golden evaluation dataset."""

    version: str
    created_at: datetime
    cases: list[GoldenCase]
    metadata: GenerationMetadata | None = None


class CaseResult(BaseModel):
    """All scoring outputs for a single case within a single run."""

    case_id: str
    prompt_version: str
    model: str
    provider: str
    predicted_category: str
    expected_category: str
    category_match: bool
    summary_score_judge: float
    summary_score_embedding: float
    summary_score_keyword: float
    composite_summary_score: float
    judge_reason: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    estimated_cost_aud: float
    run_id: str
    timestamp: datetime


DifficultyBreakdown: TypeAlias = dict[str, dict[str, float]]


class EvalRun(BaseModel):
    """Summary of a complete evaluation run across all cases."""

    run_id: str
    prompt_version: str
    model: str
    provider: str
    timestamp: datetime
    total_cases: int
    accuracy: float
    avg_composite_summary_score: float
    p95_latency_ms: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_aud: float
    status: Literal["PASS", "WARN", "FAIL"]
    difficulty_breakdown: dict[str, dict[str, float]] = Field(default_factory=dict)
    case_results: list[CaseResult] = Field(default_factory=list)


class RunDiff(BaseModel):
    """Comparison between two evaluation runs."""

    run_id: str
    prev_run_id: str | None
    accuracy_delta: float | None
    summary_score_delta: float | None
    cost_delta_aud: float | None
    regression_cases: list[str]
    improvement_cases: list[str]
    p_value: float | None
    z_statistic: float | None
    is_significant: bool
    difficulty_warnings: list[str]
    status: Literal["PASS", "WARN", "FAIL"]
