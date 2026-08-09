# LLM-CI: AI-Assisted Build Context

> This file is the single source of truth for the AI building this project.
> Do not deviate from it. On every sprint, re-read the relevant sections before writing any code.
> Never proceed to Sprint N+1 unless Sprint N is marked COMPLETED by the user.

---

## Project identity

**Name:** LLM-CI  
**Purpose:** A CI/CD-style pipeline that tests any LLM-powered feature against a golden dataset whenever a prompt or model changes, detects quality regressions across multiple providers and scoring dimensions, and alerts a Slack channel before bad outputs reach users.  
**Target user:** ML engineers and developers who ship LLM-powered features and need confidence that prompt/model changes don't silently degrade behaviour — across models, across time, and across all axes of quality.

---

## Core design principles

1. **Zero infrastructure bias.** No hosted DBs, no cloud queues. SQLite + JSON files. Everything runs locally and in GitHub Actions with no setup beyond `pip install`.
2. **Eval quality = data quality.** The golden dataset is human-curated (or carefully prompted synthetic data). The system is only as trustworthy as its ground truth.
3. **Diff-first thinking.** The point is not a score in isolation — it is the delta between runs. Every design decision serves the diff.
4. **Prompt as code.** Prompts live in versioned YAML files. Changing a prompt triggers the pipeline exactly like changing code does.
5. **Production signal over academic metrics.** Track what real teams care about: regression count, latency, token cost, slow drift, difficulty-stratified failure patterns.
6. **Stateless container.** The eval runner is a Docker container that takes env vars and produces artefacts. No hidden state.
7. **Provider-agnostic interface.** All LLM calls go through a `BaseProvider` abstract class. Adding a new provider requires implementing one method, not touching eval logic.
8. **Statistical rigour over fixed thresholds.** Regression decisions use a two-proportion z-test, not an arbitrary percentage. Fixed thresholds are a fallback, not the primary signal.
9. **Cost visibility.** Every eval run records estimated spend in AUD. Token counts and latency are first-class metrics, not footnotes.

---

## Tech stack (locked — do not change without user approval)

| Component | Tool | Notes |
|---|---|---|
| Language | Python 3.11+ | Use type hints everywhere |
| LLM under test | OpenAI `gpt-4o-mini` (default) | Swappable via provider interface; also Anthropic Claude Haiku, Google Gemini Flash |
| LLM-as-judge | OpenAI `gpt-4o-mini` | Separate call; separate prompt; provider-agnostic |
| Provider interface | `BaseProvider` ABC | `OpenAIProvider`, `AnthropicProvider`, `GeminiProvider` implement it |
| Embedding scorer | `sentence-transformers` (`all-MiniLM-L6-v2`) | Local inference; zero API cost; cosine similarity vs reference summaries |
| Statistical significance | `scipy.stats` (two-proportion z-test) | Primary regression signal; fixed thresholds are secondary fallback |
| Eval framework | Custom + `ragas` optional | Core logic is custom; ragas for supplementary metrics |
| Data storage | SQLite + JSON | SQLite for run history; JSON for golden dataset |
| Alerting | Slack Webhooks | Incoming webhook; no Slack SDK needed |
| Scheduling/CI | GitHub Actions | Triggers on `/prompts` or `/src` path changes |
| Reporting | Jinja2 HTML report + Streamlit dashboard | HTML for per-run diff; Streamlit for historical trends |
| Containerisation | Docker | Multi-stage build; env-var driven |
| Synthetic data gen | Gemini `gemini-3.5-flash-lite` | One-off script; output verified before use |
| Cost tracking | Custom (`cost.py`) | Per-run token cost in AUD; per-provider pricing constants |

---

## Repository structure (final target)

```
llm-ci/
├── CONTEXT.md                    ← This file (committed to repo)
├── README.md
├── CONTRIBUTING.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .github/
│   └── workflows/
│       ├── eval-ci.yml
│       └── tests.yml
├── prompts/
│   ├── v1.0.0.yaml
│   └── v1.1.0.yaml
├── data/
│   ├── golden_dataset.json       ← Synthetic + adversarial + multilingual, verified
│   └── golden_dataset_schema.json
├── src/
│   ├── __init__.py
│   ├── config.py                 ← Pydantic settings; reads env vars
│   ├── models.py                 ← Pydantic data models
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py               ← BaseProvider ABC
│   │   ├── openai_provider.py    ← OpenAI implementation
│   │   ├── anthropic_provider.py ← Anthropic implementation
│   │   ├── gemini_provider.py    ← Gemini implementation
│   │   └── factory.py            ← get_provider(name) → BaseProvider
│   ├── feature.py                ← The LLM feature under test (provider-agnostic)
│   ├── dataset.py                ← Dataset loader and validator
│   ├── runner.py                 ← Async eval runner
│   ├── scorer.py                 ← LLM-judge + embedding + keyword scoring
│   ├── stats.py                  ← Two-proportion z-test + significance logic
│   ├── cost.py                   ← Per-provider pricing constants + cost calculation
│   ├── comparator.py             ← Run-vs-run diff logic (uses stats.py)
│   ├── drift.py                  ← Rolling average / slow drift detection
│   ├── difficulty.py             ← Difficulty-stratified regression analysis
│   ├── storage.py                ← SQLite read/write
│   ├── reporter.py               ← HTML report generator
│   ├── alerter.py                ← Slack webhook sender
│   └── dashboard.py              ← Streamlit historical trend dashboard
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── validate_dataset.py
│   ├── test_feature.py
│   ├── test_alert.py
│   ├── demo_run.py
│   └── seed_db.py
├── templates/
│   └── report.html.j2            ← Jinja2 report template
└── tests/
    ├── test_models.py
    ├── test_scorer.py
    ├── test_stats.py
    ├── test_comparator.py
    ├── test_difficulty.py
    ├── test_cost.py
    └── test_providers.py
```

---

## The LLM feature under test

**Task:** Customer support email classifier.  
**Input:** Raw email text (string).  
**Output:** Structured JSON with two fields:
- `category`: one of `["billing", "technical", "account", "general"]`
- `summary`: one sentence (≤25 words) describing the core issue

**Prompt location:** `/prompts/v1.0.0.yaml`  
**Interface:** Single async function `classify_email(email_text: str, prompt_config: PromptConfig, provider: BaseProvider) -> ClassificationResult`  
**Provider selection:** Determined by `PROVIDER` env var. `feature.py` calls `providers/factory.py` to get the right provider instance.

---

## Provider interface specification

### `src/providers/base.py`

```python
from abc import ABC, abstractmethod
from src.models import ClassificationResult, PromptConfig

class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
    ) -> dict:
        """Make a completion call. Return dict with keys: text, input_tokens, output_tokens, latency_ms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name string e.g. 'openai', 'anthropic', 'gemini'."""
```

### `src/providers/factory.py`

```python
def get_provider(name: str, settings: Settings) -> BaseProvider:
    """Return the correct provider instance for a given name string."""
```

Supported name values: `"openai"`, `"anthropic"`, `"gemini"`.  
Default: `"openai"`.

### Provider-specific models

| Provider | Model under test | Judge model |
|---|---|---|
| openai | `gpt-4o-mini` | `gpt-4o-mini` |
| anthropic | `claude-haiku-4-5` | `claude-haiku-4-5` |
| gemini | `gemini-3.5-flash-lite` | `gemini-3.5-flash-lite` |

All three must produce identical output schema (`ClassificationResult`). Provider differences are contained entirely within `src/providers/`.

---

## Golden dataset schema

```json
{
  "version": "1.0.0",
  "created_at": "ISO8601",
  "cases": [
    {
      "id": "case_001",
      "input": "...",
      "expected_category": "billing",
      "expected_summary_keywords": ["refund", "charged"],
      "difficulty": "easy | medium | hard | edge",
      "tags": ["ambiguous", "sarcastic", "multilingual", "short", "typo"],
      "notes": "Why this case exists and what it tests"
    }
  ]
}
```

**Target size:** 100 cases. Breakdown:
- 20 billing (5 easy, 5 medium, 5 hard, 5 edge)
- 20 technical (same split)
- 20 account (same split)
- 20 general (same split)
- 10 adversarial (cross-category: deliberately crafted to fool the classifier; hand-labelled with correct category and a `adversarial_target` field naming the wrong category it's designed to trigger)
- 10 multilingual (2–3 languages other than English; correct category still determinable from content; tagged `multilingual`)

Valid tags: `["ambiguous", "sarcastic", "multilingual", "short", "typo", "emotional", "vague", "multi-issue", "adversarial", "non-english"]`

---

## Scoring dimensions (all stored per case per run)

| Dimension | Type | Method |
|---|---|---|
| `category_match` | Binary 0/1 | Exact string match |
| `summary_score_judge` | Float 1.0–5.0 | LLM-as-judge (see judge prompt below) |
| `summary_score_embedding` | Float 0.0–1.0 | Cosine similarity vs expected_summary_keywords joined as reference sentence; `all-MiniLM-L6-v2` |
| `summary_score_keyword` | Float 0.0–1.0 | Proportion of `expected_summary_keywords` present in generated summary (case-insensitive) |
| `composite_summary_score` | Float 0.0–1.0 | Weighted average: 0.5 × judge_normalised + 0.3 × embedding + 0.2 × keyword |
| `latency_ms` | Int | Wall-clock time per request |
| `input_tokens` | Int | From API response |
| `output_tokens` | Int | From API response |
| `estimated_cost_aud` | Float | Calculated by `cost.py` using provider pricing constants |

**Composite score formula:** `(judge_score / 5.0) * 0.5 + embedding_score * 0.3 + keyword_score * 0.2`  
Store all three sub-scores individually so they can be analysed separately.

---

## Regression detection (statistical significance first, fixed thresholds second)

### Primary: two-proportion z-test (`src/stats.py`)

Compare pass rates (category_match=True) between the current run and the previous run using a two-proportion z-test from `scipy.stats.proportions_ztest`. The test answers: "is this accuracy difference real, or noise from 100 cases?"

- p < 0.05 → statistically significant regression/improvement (flag it)
- p ≥ 0.05 → not significant (log it, do not alert)
- If previous run has fewer than 30 cases, fall back to fixed thresholds (insufficient sample for z-test)

Store `p_value` and `z_statistic` in every `RunDiff`.

### Secondary: fixed thresholds (fallback + additional signals)

Used when z-test is inconclusive or for metrics z-test doesn't cover:

| Metric | Warning | Critical |
|---|---|---|
| Accuracy delta (absolute) | -3% | -8% |
| Composite summary score delta | -0.1 | -0.25 |
| Regression case count | 3 cases | 8 cases |
| P95 latency increase | +500ms | +1500ms |
| Cost per run increase | +30% | +80% |

### Slow drift
Fire a warning if the 7-run moving average accuracy drops more than 5% from the window peak, even if no single run crossed thresholds. This catches gradual degradation invisible to per-run checks.

---

## Run status logic

- `PASS`: No threshold crossed
- `WARN`: One or more warning thresholds crossed, no critical
- `FAIL`: One or more critical thresholds crossed

A `FAIL` run blocks PR merge in GitHub Actions (exit code 1).

---

## Prompts

## Cost tracking specification (`src/cost.py`)

```python
PRICING_AUD = {
    "openai": {
        "gpt-4o-mini": {"input_per_1k": 0.000228, "output_per_1k": 0.000912},
        "gpt-4o":      {"input_per_1k": 0.00684,  "output_per_1k": 0.02736},
    },
    "anthropic": {
        "claude-haiku-4-5": {"input_per_1k": 0.000380, "output_per_1k": 0.001900},
    },
    "gemini": {
        "gemini-3.5-flash": {"input_per_1k": 0.002325, "output_per_1k": 0.013950},
        "gemini-3.5-flash-lite": {"input_per_1k": 0.000465, "output_per_1k": 0.003875},
    },
}
```

(Prices are approximate USD→AUD at 1.55 conversion. Update if stale. These are constants, not fetched at runtime.)

Function: `calculate_cost_aud(provider: str, model: str, input_tokens: int, output_tokens: int) -> float`  
Function: `summarise_run_cost(case_results: list[CaseResult]) -> float` — sum of all `estimated_cost_aud` values in the run.

Cost is surfaced in: the HTML report, the Slack alert, and the SQLite `eval_runs` table.

---

## Difficulty-stratified regression analysis (`src/difficulty.py`)

After every run, compute accuracy broken down by both difficulty and category. Store as a nested dict on `EvalRun`.

```python
DifficultyBreakdown = dict[str, dict[str, float]]
# e.g. {"easy": {"billing": 1.0, "technical": 0.8, ...}, "hard": {...}}
```

Function: `compute_difficulty_breakdown(case_results: list[CaseResult], dataset: GoldenDataset) -> DifficultyBreakdown`

Function: `flag_difficulty_regression(current: DifficultyBreakdown, previous: DifficultyBreakdown) -> list[str]`  
Returns a list of warning strings if hard/edge cases regress faster than easy cases (i.e., easy accuracy is stable but hard accuracy drops >10%). This is the signal that a prompt change is masking deeper failures.

Surface in: HTML report (difficulty breakdown table), Slack alert (add one line if hard/edge regression is detected).

---

## Prompts

### System prompt (v1.0.0) — stored in `/prompts/v1.0.0.yaml`

```
You are a customer support triage assistant. Your job is to classify incoming customer emails and summarise the core issue.

You must respond ONLY with a valid JSON object. No markdown, no explanation, no preamble.

The JSON must contain exactly two fields:
- "category": one of ["billing", "technical", "account", "general"]
- "summary": a single sentence of no more than 25 words describing the customer's core issue

Classification rules:
- billing: payment issues, charges, invoices, refunds, pricing questions
- technical: bugs, errors, crashes, feature not working, performance issues
- account: login, password, profile, subscription management, cancellation
- general: feature requests, compliments, vague enquiries, anything that doesn't fit the above

If the email is ambiguous between two categories, choose the one most directly actionable by a support agent.
If the email contains no clear issue, categorise as "general".
```

### LLM-as-judge prompt — stored in `src/scorer.py` as a constant

```
You are evaluating the quality of a one-sentence summary written by an AI assistant for a customer support email.

You will be given:
1. The original customer email
2. The AI-generated summary

Score the summary on a scale of 1 to 5 using these criteria:
- 5: Captures the core issue precisely, is factually accurate, reads naturally
- 4: Accurate and clear but slightly verbose or missing a minor detail
- 3: Mostly accurate but vague or imprecise in a way that could mislead a support agent
- 2: Partially accurate but missing the key issue or including irrelevant information
- 1: Inaccurate, misleading, or nonsensical

Respond ONLY with a JSON object: {"score": <integer 1-5>, "reason": "<one sentence>"}
No markdown, no preamble.

Email:
{email}

Summary:
{summary}
```

### Embedding scorer — no prompt, local model

Uses `sentence-transformers` with `all-MiniLM-L6-v2`. Loaded once at runner startup, not per-case.

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def score_embedding(generated_summary: str, reference_keywords: list[str]) -> float:
    """Cosine similarity between generated summary and keyword reference sentence."""
    reference = " ".join(reference_keywords)
    embeddings = model.encode([generated_summary, reference])
    return float(util.cos_sim(embeddings[0], embeddings[1]))
```

### Keyword scorer — no prompt, pure Python

```python
def score_keywords(generated_summary: str, expected_keywords: list[str]) -> float:
    """Proportion of expected_summary_keywords found in generated summary (case-insensitive)."""
    summary_lower = generated_summary.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in summary_lower)
    return matches / len(expected_keywords) if expected_keywords else 0.0
```

### Synthetic data generation prompt (standard cases) — stored in `scripts/generate_synthetic_data.py`

```
You are generating a diverse, realistic golden dataset for evaluating a customer support email classifier.

Generate {count} customer support emails for the category: {category}
Difficulty level: {difficulty}

Difficulty definitions:
- easy: Clear, unambiguous emails where the category is obvious
- medium: Realistic emails with some ambiguity or mixed content
- hard: Emails where the correct category requires inference; could plausibly be another category
- edge: Deliberately tricky — sarcastic, very short, typos, mixed languages, no clear issue, or emotionally charged

For each email, respond ONLY with a JSON array. Each object must have:
- "input": the raw email text (realistic, varied length, no template feel)
- "expected_category": "{category}"
- "expected_summary_keywords": array of 2-4 keywords a good summary should contain
- "difficulty": "{difficulty}"
- "tags": array of relevant tags from ["ambiguous", "sarcastic", "multilingual", "short", "typo", "emotional", "vague", "multi-issue"]
- "notes": one sentence explaining what makes this case useful for evaluation

Rules:
- Vary email length: some 1–2 sentences, some 3–4 paragraphs
- Vary formality: casual, professional, frustrated, polite
- Do NOT use placeholder names like [Name] — invent realistic names
- Do NOT start every email with "I am writing to"
- Make 20% of emails contain at least one typo or grammar error
- For edge cases: at least one email should be entirely in a language other than English

No markdown. No preamble. Output only the JSON array.
```

### Adversarial case generation prompt — stored in `scripts/generate_synthetic_data.py`

```
You are generating adversarial test cases for a customer support email classifier.

An adversarial case is an email that a poorly-tuned classifier will likely misclassify, but which has a clear correct label when read carefully.

Generate {count} adversarial emails. Each email should be designed to trick the classifier into predicting {wrong_category}, but the correct label is {correct_category}.

For each email, respond ONLY with a JSON array. Each object must have:
- "input": the raw email text
- "expected_category": "{correct_category}"
- "adversarial_target": "{wrong_category}" (the category it is designed to fool the classifier into predicting)
- "expected_summary_keywords": array of 2-4 keywords a good summary should contain
- "difficulty": "edge"
- "tags": ["adversarial"]
- "notes": one sentence explaining the specific linguistic trick used (e.g., "uses billing vocabulary while describing a login issue")

Rules:
- The trick must be linguistic, not just ambiguous — there should be a clear correct answer
- Vary the technique: vocabulary mismatch, topic switching mid-email, misleading subject line references, category-adjacent terminology
- Do NOT invent fake product features — keep emails plausibly real

No markdown. No preamble. Output only the JSON array.
```

### Multilingual case generation prompt — stored in `scripts/generate_synthetic_data.py`

```
You are generating multilingual test cases for a customer support email classifier that should work regardless of language.

Generate {count} customer support emails. Write each email in a different non-English language. Distribute across: Spanish, French, Hindi, Japanese, Portuguese, Arabic (pick the most varied set for the count requested).

The correct category for each email is {category}.

For each email, respond ONLY with a JSON array. Each object must have:
- "input": the raw email text in the target language (do NOT include a translation)
- "expected_category": "{category}"
- "expected_summary_keywords": array of 2-4 keywords IN ENGLISH that a good summary should contain
- "difficulty": "edge"
- "tags": ["multilingual", "non-english"]
- "notes": one sentence in English explaining what the email says and why this category is correct

No markdown. No preamble. Output only the JSON array.
```

### Slack alert message format

```
{
  "blocks": [
    {
      "type": "header",
      "text": {"type": "plain_text", "text": "LLM-CI Eval Run: {STATUS}"}
    },
    {
      "type": "section",
      "fields": [
        {"type": "mrkdwn", "text": "*Prompt version:*\n{prompt_version}"},
        {"type": "mrkdwn", "text": "*Provider / Model:*\n{provider} / {model}"},
        {"type": "mrkdwn", "text": "*Accuracy:*\n{accuracy}% (was {prev_accuracy}%)"},
        {"type": "mrkdwn", "text": "*Significance:*\np={p_value} ({significant})"},
        {"type": "mrkdwn", "text": "*Regressions:*\n{regression_count} cases"},
        {"type": "mrkdwn", "text": "*Est. cost:*\nAUD ${run_cost} (was ${prev_cost})"},
        {"type": "mrkdwn", "text": "*Run ID:*\n{run_id}"}
      ]
    },
    {
      "type": "section",
      "text": {"type": "mrkdwn", "text": "{regression_summary}"}
    },
    {
      "type": "section",
      "text": {"type": "mrkdwn", "text": "{difficulty_warning}"}
    },
    {
      "type": "actions",
      "elements": [
        {
          "type": "button",
          "text": {"type": "plain_text", "text": "View Full Report"},
          "url": "{report_url}"
        }
      ]
    }
  ]
}
```

`{significant}` renders as `"significant ⚠️"` or `"not significant"`.  
`{difficulty_warning}` is empty string if no difficulty regression; otherwise `"⚠️ Hard/edge cases regressing faster than easy cases"`.  
`{p_value}` rounded to 3 decimal places.

---

## GitHub Actions workflow specification

**File:** `.github/workflows/eval-ci.yml`  
**Trigger:** Push or PR that modifies any file under `prompts/` or `src/`  
**Steps:**
1. Checkout repo
2. Set up Python 3.11
3. Install dependencies from `requirements.txt`
4. Run `python -m src.runner --prompt prompts/$(ls -t prompts/ | head -1) --output results/`
5. Run `python -m src.comparator --current results/latest.json --baseline results/baseline.json`
6. Run `python -m src.reporter --run-id ${{ github.run_id }}`
7. Upload HTML report as GitHub Actions artefact
8. Post PR comment with summary (use `actions/github-script`)
9. Exit 1 if status is FAIL

**Required secrets:** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `SLACK_WEBHOOK_URL`  
**Optional:** `PROVIDER` (defaults to `openai`), `MODEL_UNDER_TEST` (defaults to provider default)

---

## Sprint map (do not skip ahead)

| Sprint | Focus | Gate condition |
|---|---|---|
| S1 | Scaffolding + models + config | All Pydantic models pass validation; directory structure matches spec |
| S2 | Provider interface + feature function | All three providers return valid `ClassificationResult`; factory selects correctly |
| S3 | Prompt YAML system | `classify_email()` works with OpenAI for 3 test inputs; v1.0.0 and v1.1.0 prompts load |
| S4 | Synthetic dataset (standard + adversarial + multilingual) | 100 cases generated, schema-validated; all tags and difficulties present |
| S5 | Eval runner (async, batched) | Runner processes all 100 cases; results written to SQLite with cost fields |
| S6 | Scoring engine (judge + embedding + keyword + composite) | All scoring dimensions populated per case; `tests/test_scorer.py` passes |
| S7 | Stats + comparator + difficulty analysis | z-test runs; difficulty breakdown computed; `tests/test_stats.py` and `test_comparator.py` pass |
| S8 | Drift detection + cost tracking | Slow drift fires correctly; cost per run stored and summed |
| S9 | HTML reporter + Slack alerter | Report renders with all sections; Slack alert sends with p-value and cost |
| S10 | Streamlit dashboard + GitHub Actions + Docker | Dashboard shows trend charts; pipeline runs in container; PR gate works |
| S11 | Polish + README + Loom outline | Fresh clone runs in under 10 minutes; portfolio artefacts ready |

---

## AI behaviour rules (critical — read before every sprint)

1. **One sprint at a time.** Write all code for the current sprint. Then stop. Do not write code for future sprints.
2. **Follow the file structure exactly.** Every file goes where the structure above says it goes.
3. **Read this context file at the start of every sprint.** Do not rely on memory.
4. **When a function signature is specified here, use it exactly.** Do not rename, restructure, or "improve" the interface without flagging it first.
5. **All prompts are defined above.** Do not invent or modify prompts. Use them verbatim.
6. **Type everything.** No `Any`, no bare `dict`. Pydantic models for all structured data.
7. **Error handling is not optional.** Every API call wrapped in try/except with structured logging.
8. **Do not use print() for status output.** Use Python's `logging` module configured at `INFO` level.
9. **Every function gets a one-line docstring.** No more, no less.
10. **At end of each sprint, output a checklist of what was built and a one-line test command the user can run to verify the gate condition.**

---

## Environment variables (complete list)

```
# Provider API keys
OPENAI_API_KEY=             # Required if PROVIDER=openai (default)
ANTHROPIC_API_KEY=          # Required if PROVIDER=anthropic
GEMINI_API_KEY=             # Required if PROVIDER=gemini

# Provider selection
PROVIDER=openai                       # Default; options: openai, anthropic, gemini
MODEL_UNDER_TEST=gpt-4o-mini          # Default (provider-specific; overrides provider default)
JUDGE_PROVIDER=openai                 # Default; judge can differ from model under test
JUDGE_MODEL=gpt-4o-mini               # Default
GEMINI_REQUESTS_PER_MINUTE=14         # Default; paces Gemini calls under the free-tier 15 RPM cap

# Alerting
SLACK_WEBHOOK_URL=          # Required for alerting

# Statistical significance
SIGNIFICANCE_ALPHA=0.05               # Default p-value threshold for z-test
MIN_CASES_FOR_ZTEST=30                # Default; fall back to fixed thresholds below this

# Fixed threshold fallbacks
WARN_ACCURACY_DELTA=-0.03             # Default
CRITICAL_ACCURACY_DELTA=-0.08         # Default
WARN_REGRESSION_COUNT=3               # Default
CRITICAL_REGRESSION_COUNT=8           # Default
WARN_SUMMARY_DELTA=-0.1               # Default (composite score)
CRITICAL_SUMMARY_DELTA=-0.25          # Default (composite score)
WARN_LATENCY_INCREASE_MS=500          # Default
CRITICAL_LATENCY_INCREASE_MS=1500     # Default
WARN_COST_INCREASE_PCT=0.30           # Default
CRITICAL_COST_INCREASE_PCT=0.80       # Default

# Drift
SLOW_DRIFT_WINDOW=7                   # Default (runs)
SLOW_DRIFT_THRESHOLD=-0.05            # Default

# Embedding scorer
EMBEDDING_MODEL=all-MiniLM-L6-v2      # Default; local sentence-transformers model

# Paths
DB_PATH=data/runs.db                  # Default
GOLDEN_DATASET_PATH=data/golden_dataset.json   # Default
PROMPTS_DIR=prompts/                  # Default
RESULTS_DIR=results/                  # Default
LOG_LEVEL=INFO                        # Default

# Dashboard
DASHBOARD_PORT=8501                   # Default Streamlit port
```

---

## What DONE looks like (portfolio standard)

- A developer clones the repo, adds `OPENAI_API_KEY` and `SLACK_WEBHOOK_URL` to `.env`, runs `docker-compose up`, and sees an eval run complete with a Slack alert, HTML diff report, and cost summary — in under 10 minutes.
- Changing a word in a prompt YAML triggers CI, runs statistical significance testing against the previous run, and blocks merge if the regression is real (p < 0.05 and accuracy drops > 8%).
- The Streamlit dashboard opens at `localhost:8501` and shows accuracy, cost, and composite summary score trends across all stored runs, with a difficulty breakdown table.
- Swapping `PROVIDER=anthropic` in `.env` and re-running uses Claude Haiku with zero other changes.
- The README explains what this is, why it matters, and how to add a test case — in under 500 words.
- There is a 3-minute Loom showing: prompt change → CI trigger → regression detected (with p-value) → diff report → Slack alert → difficulty breakdown flagging hard cases regressing.
