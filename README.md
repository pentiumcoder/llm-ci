# LLM-CI

**A CI/CD pipeline for testing LLM-powered features — prompt as code, statistics-first regression detection, cost visibility, and Slack alerts.**

![Tests](https://github.com/pentiumcoder/llm-ci/actions/workflows/tests.yml/badge.svg)
![Eval CI](https://github.com/pentiumcoder/llm-ci/actions/workflows/eval-ci.yml/badge.svg)

Whenever a prompt, model, or provider changes, LLM-CI re-runs the feature against a 100-case golden dataset, compares the result to the previous run, and answers one question: **is this a real regression or just noise?** If a change degrades quality in a statistically significant way, the pipeline blocks the merge and alerts Slack before bad outputs reach users.

---

## What it is

LLM-CI treats a production prompt like code:

- **Prompts are versioned YAML files** in `prompts/`. Changing a prompt triggers the pipeline exactly like changing code triggers a build.
- **Every run is scored on 9 dimensions** — category accuracy, LLM-judge summary quality, embedding similarity, keyword recall, latency, token usage, and estimated cost in AUD.
- **Regressions are decided by a two-proportion z-test**, not an arbitrary percentage. This distinguishes a 2% drop that's noise (100 cases) from a 2% drop that's real (10,000 cases).
- **Runs are stored in SQLite** and rendered as self-contained HTML diff reports, so every prompt change has an auditable quality history.
- **Slack alerts** fire on every eval with status, accuracy, p-value, regression count, and cost — with a link to the full report.

## The feature under test

The project evaluates a **customer support email classifier**:

> **Input:** raw customer email &nbsp;·&nbsp; **Output:** `{ "category": "billing" | "technical" | "account" | "general", "summary": "<one sentence, ≤25 words>" }`

Three LLM providers implement a single `BaseProvider` interface and are fully interchangeable:

| Provider | Model under test / judge | Notes |
|---|---|---|
| Google Gemini | `gemini-3.5-flash-lite` | Free tier, rate-limited at 14 RPM; recommended default |
| OpenAI | `gpt-4o-mini` | |
| Anthropic | `claude-haiku-4-5` | |

## How the pipeline works

```
prompt change ──► GitHub Actions ──► async runner ──► classifier (100 cases)
                                                        │
Golden dataset (data/golden_dataset.json) ◄─────────────┘
                                                        │
                              scorer (judge + embedding + keyword + cost)
                                                        │
                        comparator (two-proportion z-test vs previous run)
                                                        │
                   PASS / WARN / FAIL ──► HTML diff report + SQLite + Slack
```

1. **Load** the versioned prompt + the 100-case golden dataset.
2. **Classify** all cases concurrently (5-way, rate-limiter-pacing LLM calls).
3. **Score** each summary: LLM-as-judge (1–5), local embedding cosine similarity, keyword recall, composite score.
4. **Compare** against the previous run in the database with a two-proportion z-test plus fixed-threshold fallbacks.
5. **Report**: a self-contained HTML diff report, a row in SQLite, and a Slack alert. `FAIL` exits 1 and blocks the PR.

### Scoring dimensions

| Dimension | Method |
|---|---|
| `category_match` | Exact category string match |
| `summary_score_judge` | LLM-as-judge, 1.0–5.0 |
| `summary_score_embedding` | Cosine similarity vs keyword reference (`all-MiniLM-L6-v2`, local) |
| `summary_score_keyword` | Proportion of expected keywords in the summary |
| `composite_summary_score` | 0.5 × judge + 0.3 × embedding + 0.2 × keyword |
| `latency_ms`, `input_tokens`, `output_tokens` | Per-request telemetry |
| `estimated_cost_aud` | Provider pricing constants → AUD |

### Regression detection (statistics first)

- **Primary:** two-proportion z-test (`scipy`) comparing pass rates between runs. `p < 0.05` → statistically significant; `p ≥ 0.05` → noise, no alert.
- **Fallback:** fixed thresholds when the sample is too small (<30 cases): warning at −3% accuracy, critical at −8%, plus regression-count, composite-score, latency, and cost deltas.
- **Slow drift:** a 7-run moving average dropping >5% from its peak fires a warning even if no single run crosses a threshold.
- **Difficulty stratification:** accuracy is broken down by difficulty × category, so a prompt silently overfitting easy cases while hard/edge cases collapse is flagged explicitly.

## The golden dataset

100 human-curated (LLM-generated, schema-validated) cases — **20 easy, 20 medium, 20 hard, 40 edge** — covering all four categories, plus deliberately nasty content:

- **10 adversarial** cases engineered to trick a naive classifier (vocabulary mismatches, topic switches, misleading wording) with a known correct label
- **15 multilingual** cases in Spanish, French, Hindi, Japanese, Portuguese, and Arabic
- Typo-ridden, sarcastic, emotionally charged, vague, and multi-issue emails

Dataset metadata (`data/golden_dataset.json`): generated by `gemini-3.5-flash-lite`, 25 API calls, ~AUD $0.06.

## Results

Real recorded runs from `data/runs.db` (Gemini free tier, `gemini-3.5-flash-lite`, 100 cases each):

| Run | Prompt | Accuracy | Composite | P95 latency | Cost (AUD) | Status |
|---|---|---|---|---|---|---|
| `29198bee` | v1.0.0 | 75.0% | 0.689 | 1281 ms | $0.024 | PASS |
| `050fb277` | v1.0.0 | 76.0% | 0.713 | 1392 ms | $0.024 | PASS |
| `c37fbea7` | v1.1.0 | 77.0% | 0.677 | 1219 ms | $0.025 | PASS |
| `28a5c894` | v1.1.0 | 74.0% | 0.709 | 1220 ms | $0.025 | PASS |

**Headline regression gate — v1.1.0 vs v1.0.0 (run `28a5c894`):**

```
Accuracy:     74.0%   (previous 76.0%)   delta −2.0%
Z-statistic: −0.327    P-value: 0.744     significant? No
Status:       PASS    — the 2% drop is noise at n=100, not a regression
```

**What it costs:** roughly **AUD $0.025 (~1.6 US cents) per 100-case eval** on free-tier Gemini — a full quality gate for less than a cup of coffee.

**What difficulty stratification caught** (run `28a5c894`):

| Difficulty | Accuracy |
|---|---|
| Easy | 85% |
| Hard | 85% |
| Medium | 75% |
| Edge (typos, sarcasm, multilingual, adversarial) | **62%** |

Edge cases are the known weak spot — exactly the signal the difficulty breakdown exists to surface. Aggregate accuracy (~74%) hides that adversarial and multilingual emails fail ~1.5× more often than routine ones.

## Architecture decisions

- **Provider interface (ABC pattern).** All LLM calls go through `BaseProvider.complete()`. Adding a provider means implementing one async method — runner, scorer, and reporter never change. Provider quirks (rate limits, response formats, retry semantics) live inside `src/providers/`.
- **Statistical rigour over fixed thresholds.** The z-test prevents false alarms on small runs and missed regressions on large ones; fixed thresholds are the fallback.
- **Difficulty stratification.** "Overall accuracy dropped 2%" is less actionable than "hard cases regressed 12% while easy cases stayed stable."
- **Zero infrastructure bias.** SQLite + JSON + local embedding model. No hosted DBs, no cloud queues — `pip install` and go.
- **Resilience by design.** Free-tier LLMs fail: the runner retries 429s (rate limit) and 5xx "high demand" spikes with exponential backoff, and a case that ultimately fails degrades to a zero-scoring row instead of aborting the run — so a full eval always completes, persists, and alerts.

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ (fully typed, Pydantic models) |
| LLM providers | OpenAI, Anthropic, Google Gemini (`google-genai` SDK) |
| Embedding scorer | `sentence-transformers` (`all-MiniLM-L6-v2`) — local, zero API cost |
| Statistics | `scipy` two-proportion z-test |
| Storage | SQLite (`data/runs.db`) + JSON golden dataset |
| Reporting | Jinja2 HTML diff reports + Streamlit trend dashboard |
| Alerting | Slack incoming webhooks (no SDK) |
| CI/CD | GitHub Actions (`tests.yml`, `eval-ci.yml`) |
| Containerisation | Multi-stage Dockerfile + docker-compose |

## Project structure

```
llm-ci/
├── prompts/                # versioned prompt YAML files (v1.0.0, v1.1.0)
├── data/                   # golden dataset (100 cases) + run history (SQLite)
├── src/
│   ├── providers/          # BaseProvider ABC + openai/anthropic/gemini
│   ├── feature.py          # the classifier under test
│   ├── runner.py           # async eval runner (5-way concurrency)
│   ├── scorer.py           # judge + embedding + keyword scoring
│   ├── comparator.py       # z-test run-vs-run diff
│   ├── stats.py, difficulty.py, drift.py, cost.py
│   ├── storage.py, reporter.py, alerter.py, dashboard.py
│   └── config.py           # all settings, env-driven
├── scripts/                # dataset generation, validation, demo, seed
├── templates/report.html.j2
├── results/                # committed HTML diff reports
└── tests/                  # 133 tests
```

## Quickstart

```bash
git clone https://github.com/pentiumcoder/llm-ci.git
cd llm-ci
pip install -r requirements.txt
```

Create `.env` (see `src/config.py` for the full list):

```
PROVIDER=gemini
GEMINI_API_KEY=your_key
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

Run the pipeline:

```bash
python -m src.runner --prompt prompts/v1.1.0.yaml
streamlit run src/dashboard.py          # trend dashboard at localhost:8501
python scripts/test_alert.py            # verify your Slack webhook
```

Or with Docker:

```bash
docker-compose up eval dashboard
```

## Adding a test case

Edit `data/golden_dataset.json` → add an object to `cases` with `id`, `input`, `expected_category`, `expected_summary_keywords`, `difficulty`, `tags`, `notes` (and optionally `adversarial_target`). Validate:

```bash
python scripts/validate_dataset.py
```

## Switching providers

Set `PROVIDER=anthropic` or `PROVIDER=openai` in `.env`. The factory selects the provider class automatically — no code changes. Judge and model-under-test can differ (`JUDGE_PROVIDER`, `JUDGE_MODEL`).

## Adjusting thresholds

All tunable via env vars with defaults: `SIGNIFICANCE_ALPHA=0.05`, `WARN_ACCURACY_DELTA=-0.03`, `CRITICAL_ACCURACY_DELTA=-0.08`, `SLOW_DRIFT_WINDOW=7`, `GEMINI_REQUESTS_PER_MINUTE=14`, and more — see `src/config.py`.

## CI

- **`tests.yml`** — runs the 133-test suite on every push.
- **`eval-ci.yml`** — runs a full live eval on every push touching `prompts/` or `src/` (and via `workflow_dispatch`), uploads the HTML report as an artefact, and **fails the run if the eval status is FAIL** — a hard merge gate driven by real model quality, not just unit tests.

## Tests

```bash
python -m pytest tests/ -q       # 133 passing
```

Covers providers (incl. 429/5xx retry behaviour, rate limiting), scoring, statistics, comparator, difficulty analysis, cost, and runner failure tolerance.

---

An end-to-end LLM evaluation platform. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines.
