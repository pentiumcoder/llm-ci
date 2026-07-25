# LLM-CI

CI/CD pipeline that tests LLM-powered features against a 100-case golden dataset on every prompt or model change, detects quality regressions using statistical significance testing, and alerts Slack before bad outputs reach users.

## Setup

1. Clone the repo and `cd llm-ci`
2. Create `.env` with `OPENAI_API_KEY` and `SLACK_WEBHOOK_URL`
3. `pip install -r requirements.txt`
4. `python -m src.runner --prompt prompts/v1.0.0.yaml` — runs the eval pipeline
5. Open `results/report_*.html` for the diff report
6. `python scripts/test_alert.py` — sends a synthetic Slack alert to verify your webhook
7. `streamlit run src/dashboard.py` — opens the historical trend dashboard at localhost:8501
8. For Docker: `docker-compose up eval` runs the pipeline; `docker-compose up dashboard` serves the dashboard on port 8501

## Adding a test case

Edit `data/golden_dataset.json`. Add an object to the `cases` array with these fields: `id`, `input`, `expected_category` (one of billing/technical/account/general), `expected_summary_keywords` (2–4 words), `difficulty` (easy/medium/hard/edge), `tags`, `notes`, and optionally `adversarial_target`. Run `python scripts/validate_dataset.py` to verify schema compliance.

## Switching providers

Set `PROVIDER=anthropic` (or `gemini`) in `.env`. The runner automatically selects the correct provider class via `src/providers/factory.py`. No code changes needed. Each provider maps to a specific model: OpenAI → `gpt-4o-mini`, Anthropic → `claude-haiku-4-5`, Gemini → `gemini-2.0-flash`.

## Adjusting thresholds

All thresholds are configurable via environment variables with sensible defaults. Key ones: `WARN_ACCURACY_DELTA=-0.03`, `CRITICAL_ACCURACY_DELTA=-0.08`, `SIGNIFICANCE_ALPHA=0.05`, `SLOW_DRIFT_WINDOW=7`, `SLOW_DRIFT_THRESHOLD=-0.05`. See `src/config.py` for the complete list.

## Architecture decisions

**Provider interface (ABC pattern).** All LLM calls go through `BaseProvider.complete()`. Adding a new provider means implementing one async method — the eval runner, scorer, and reporter never change. This contains provider-specific quirks (rate limits, response formats) inside `src/providers/` where they belong.

**Two-proportion z-test over fixed thresholds.** A 2% accuracy drop on 100 cases is noise; a 2% drop on 10,000 cases is real. The z-test answers "is this difference statistically significant given our sample size?" Fixed thresholds (±3% warning, ±8% critical) are a fallback when sample sizes are too small for the test, not the primary signal. This prevents false alarms on small runs and missed regressions on large ones.

**Difficulty stratification.** Easy cases catching 100% while hard cases drop 15% is a signal that the prompt is overfitting to simple patterns. By breaking down accuracy by difficulty level (easy/medium/hard/edge), we surface regressions that aggregate accuracy numbers hide. This is the difference between "overall accuracy dropped 2%" and "hard cases regressed 12% while easy cases stayed stable."
