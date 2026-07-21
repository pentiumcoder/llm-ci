"""Tests for all Pydantic data models in src.models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.models import (
    CaseResult,
    ClassificationResult,
    GoldenCase,
    GoldenDataset,
    EvalRun,
    PromptConfig,
    RunDiff,
)


# ---------------------------------------------------------------------------
# PromptConfig
# ---------------------------------------------------------------------------

class TestPromptConfig:
    """Tests for the PromptConfig model."""

    def test_valid(self):
        cfg = PromptConfig(
            version="1.0.0",
            created_at=datetime.now(tz=timezone.utc),
            system_prompt="You are a classifier.",
        )
        assert cfg.version == "1.0.0"
        assert cfg.few_shot_examples == []

    def test_missing_version(self):
        with pytest.raises(ValidationError):
            PromptConfig(
                created_at=datetime.now(tz=timezone.utc),
                system_prompt="prompt",
            )

    def test_missing_system_prompt(self):
        with pytest.raises(ValidationError):
            PromptConfig(
                version="1.0.0",
                created_at=datetime.now(tz=timezone.utc),
            )


# ---------------------------------------------------------------------------
# ClassificationResult
# ---------------------------------------------------------------------------

class TestClassificationResult:
    """Tests for the ClassificationResult model."""

    def test_valid(self):
        cr = ClassificationResult(
            category="billing",
            summary="Customer was double-charged for their subscription.",
            input_tokens=120,
            output_tokens=30,
            latency_ms=850,
            provider="openai",
        )
        assert cr.category == "billing"
        assert cr.estimated_cost_aud == 0.0

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            ClassificationResult(
                category="shipping",
                summary="Something",
                input_tokens=10,
                output_tokens=5,
                latency_ms=100,
                provider="openai",
            )

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            ClassificationResult(category="general")


# ---------------------------------------------------------------------------
# GoldenCase
# ---------------------------------------------------------------------------

class TestGoldenCase:
    """Tests for the GoldenCase model."""

    def test_valid(self):
        gc = GoldenCase(
            id="case_001",
            input="Where is my refund?",
            expected_category="billing",
            expected_summary_keywords=["refund", "missing"],
            difficulty="easy",
            tags=["short"],
            notes="Simple refund question.",
        )
        assert gc.id == "case_001"
        assert gc.adversarial_target is None

    def test_invalid_difficulty(self):
        with pytest.raises(ValidationError):
            GoldenCase(
                id="case_001",
                input="Hello",
                expected_category="general",
                expected_summary_keywords=[],
                difficulty="impossible",
                tags=[],
                notes="test",
            )

    def test_adversarial_target_present(self):
        gc = GoldenCase(
            id="case_adv_01",
            input="I keep getting charged extra.",
            expected_category="technical",
            expected_summary_keywords=["charge", "extra"],
            difficulty="edge",
            tags=["adversarial"],
            notes="Uses billing vocab for a technical issue.",
            adversarial_target="billing",
        )
        assert gc.adversarial_target == "billing"


# ---------------------------------------------------------------------------
# GoldenDataset
# ---------------------------------------------------------------------------

class TestGoldenDataset:
    """Tests for the GoldenDataset model."""

    def test_valid(self):
        ds = GoldenDataset(
            version="1.0.0",
            created_at=datetime.now(tz=timezone.utc),
            cases=[],
        )
        assert ds.version == "1.0.0"
        assert len(ds.cases) == 0

    def test_with_cases(self):
        case = GoldenCase(
            id="c1",
            input="Help",
            expected_category="general",
            expected_summary_keywords=["help"],
            difficulty="easy",
            tags=[],
            notes="test",
        )
        ds = GoldenDataset(
            version="1.0.0",
            created_at=datetime.now(tz=timezone.utc),
            cases=[case],
        )
        assert len(ds.cases) == 1


# ---------------------------------------------------------------------------
# CaseResult
# ---------------------------------------------------------------------------

class TestCaseResult:
    """Tests for the CaseResult model."""

    def _make(self, **overrides):
        defaults = dict(
            case_id="case_001",
            prompt_version="1.0.0",
            model="gpt-4o-mini",
            provider="openai",
            predicted_category="billing",
            expected_category="billing",
            category_match=True,
            summary_score_judge=4.5,
            summary_score_embedding=0.85,
            summary_score_keyword=1.0,
            composite_summary_score=0.91,
            judge_reason="Accurate and concise.",
            latency_ms=780,
            input_tokens=120,
            output_tokens=30,
            estimated_cost_aud=0.0012,
            run_id="run_001",
            timestamp=datetime.now(tz=timezone.utc),
        )
        defaults.update(overrides)
        return CaseResult(**defaults)

    def test_valid(self):
        cr = self._make()
        assert cr.case_id == "case_001"
        assert cr.category_match is True

    def test_missing_case_id(self):
        with pytest.raises(ValidationError):
            self._make(case_id=None)

    def test_missing_predicted_category(self):
        with pytest.raises(ValidationError):
            self._make(predicted_category=None)

    def test_missing_latency_ms(self):
        with pytest.raises(ValidationError):
            self._make(latency_ms=None)

    def test_missing_run_id(self):
        with pytest.raises(ValidationError):
            self._make(run_id=None)


# ---------------------------------------------------------------------------
# EvalRun
# ---------------------------------------------------------------------------

class TestEvalRun:
    """Tests for the EvalRun model."""

    def _make(self, **overrides):
        defaults = dict(
            run_id="run_001",
            prompt_version="1.0.0",
            model="gpt-4o-mini",
            provider="openai",
            timestamp=datetime.now(tz=timezone.utc),
            total_cases=100,
            accuracy=0.85,
            avg_composite_summary_score=0.78,
            p95_latency_ms=1200,
            total_input_tokens=15000,
            total_output_tokens=4000,
            total_cost_aud=0.45,
            status="PASS",
        )
        defaults.update(overrides)
        return EvalRun(**defaults)

    def test_valid(self):
        er = self._make()
        assert er.run_id == "run_001"
        assert er.difficulty_breakdown == {}
        assert er.case_results == []

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            self._make(status="UNKNOWN")

    def test_missing_run_id(self):
        with pytest.raises(ValidationError):
            self._make(run_id=None)

    def test_missing_total_cases(self):
        with pytest.raises(ValidationError):
            self._make(total_cases=None)


# ---------------------------------------------------------------------------
# RunDiff
# ---------------------------------------------------------------------------

class TestRunDiff:
    """Tests for the RunDiff model."""

    def _make(self, **overrides):
        defaults = dict(
            run_id="run_002",
            prev_run_id="run_001",
            accuracy_delta=-0.05,
            summary_score_delta=-0.08,
            cost_delta_aud=0.01,
            regression_cases=["case_010", "case_022"],
            improvement_cases=["case_005"],
            p_value=0.03,
            z_statistic=-2.15,
            is_significant=True,
            difficulty_warnings=[],
            status="WARN",
        )
        defaults.update(overrides)
        return RunDiff(**defaults)

    def test_valid(self):
        rd = self._make()
        assert rd.is_significant is True
        assert rd.status == "WARN"

    def test_no_previous_run(self):
        rd = self._make(prev_run_id=None, p_value=None, z_statistic=None)
        assert rd.prev_run_id is None

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            self._make(status="PASSING")

    def test_missing_run_id(self):
        with pytest.raises(ValidationError):
            self._make(run_id=None)
