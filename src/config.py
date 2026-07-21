"""Pydantic Settings class that reads all environment variables for LLM-CI."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for LLM-CI. Reads from environment variables with defaults."""

    # Provider API keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    # Provider selection
    provider: str = "openai"
    model_under_test: str = "gpt-4o-mini"
    judge_provider: str = "openai"
    judge_model: str = "gpt-4o-mini"

    # Alerting
    slack_webhook_url: str = ""

    # Statistical significance
    significance_alpha: float = 0.05
    min_cases_for_ztest: int = 30

    # Fixed threshold fallbacks
    warn_accuracy_delta: float = -0.03
    critical_accuracy_delta: float = -0.08
    warn_regression_count: int = 3
    critical_regression_count: int = 8
    warn_summary_delta: float = -0.1
    critical_summary_delta: float = -0.25
    warn_latency_increase_ms: int = 500
    critical_latency_increase_ms: int = 1500
    warn_cost_increase_pct: float = 0.30
    critical_cost_increase_pct: float = 0.80

    # Drift
    slow_drift_window: int = 7
    slow_drift_threshold: float = -0.05

    # Embedding scorer
    embedding_model: str = "all-MiniLM-L6-v2"

    # Paths
    db_path: str = "data/runs.db"
    golden_dataset_path: str = "data/golden_dataset.json"
    prompts_dir: str = "prompts/"
    results_dir: str = "results/"
    log_level: str = "INFO"

    # Dashboard
    dashboard_port: int = 8501

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
