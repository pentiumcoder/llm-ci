"""SQLite read/write for eval run persistence."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime

from src.models import CaseResult, EvalRun

logger = logging.getLogger(__name__)

_CREATE_EVAL_RUNS = """
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id TEXT PRIMARY KEY,
    prompt_version TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    total_cases INTEGER NOT NULL,
    accuracy REAL NOT NULL,
    avg_composite_summary_score REAL NOT NULL,
    p95_latency_ms INTEGER NOT NULL,
    total_input_tokens INTEGER NOT NULL,
    total_output_tokens INTEGER NOT NULL,
    total_cost_aud REAL NOT NULL,
    status TEXT NOT NULL,
    difficulty_breakdown TEXT NOT NULL DEFAULT '{}',
    case_results_json TEXT NOT NULL DEFAULT '[]'
);
"""

_CREATE_CASE_RESULTS = """
CREATE TABLE IF NOT EXISTS case_results (
    case_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    predicted_category TEXT NOT NULL,
    expected_category TEXT NOT NULL,
    category_match INTEGER NOT NULL,
    summary_score_judge REAL NOT NULL,
    summary_score_embedding REAL NOT NULL,
    summary_score_keyword REAL NOT NULL,
    composite_summary_score REAL NOT NULL,
    judge_reason TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    estimated_cost_aud REAL NOT NULL,
    timestamp TEXT NOT NULL,
    PRIMARY KEY (case_id, run_id),
    FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
);
"""


def init_db(db_path: str) -> None:
    """Create eval_runs and case_results tables if they do not exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute(_CREATE_EVAL_RUNS)
        conn.execute(_CREATE_CASE_RESULTS)
        conn.commit()
        logger.info("Database initialised at %s", db_path)
    finally:
        conn.close()


def save_run(run: EvalRun, db_path: str) -> None:
    """Persist an EvalRun and its CaseResults to SQLite."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute(
            """INSERT OR REPLACE INTO eval_runs
               (run_id, prompt_version, model, provider, timestamp,
                total_cases, accuracy, avg_composite_summary_score,
                p95_latency_ms, total_input_tokens, total_output_tokens,
                total_cost_aud, status, difficulty_breakdown, case_results_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run.run_id,
                run.prompt_version,
                run.model,
                run.provider,
                run.timestamp.isoformat(),
                run.total_cases,
                run.accuracy,
                run.avg_composite_summary_score,
                run.p95_latency_ms,
                run.total_input_tokens,
                run.total_output_tokens,
                run.total_cost_aud,
                run.status,
                json.dumps(run.difficulty_breakdown),
                json.dumps([cr.model_dump(mode="json") for cr in run.case_results]),
            ),
        )

        for cr in run.case_results:
            conn.execute(
                """INSERT OR REPLACE INTO case_results
                   (case_id, run_id, prompt_version, model, provider,
                    predicted_category, expected_category, category_match,
                    summary_score_judge, summary_score_embedding,
                    summary_score_keyword, composite_summary_score,
                    judge_reason, latency_ms, input_tokens, output_tokens,
                    estimated_cost_aud, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cr.case_id,
                    cr.run_id,
                    cr.prompt_version,
                    cr.model,
                    cr.provider,
                    cr.predicted_category,
                    cr.expected_category,
                    int(cr.category_match),
                    cr.summary_score_judge,
                    cr.summary_score_embedding,
                    cr.summary_score_keyword,
                    cr.composite_summary_score,
                    cr.judge_reason,
                    cr.latency_ms,
                    cr.input_tokens,
                    cr.output_tokens,
                    cr.estimated_cost_aud,
                    cr.timestamp.isoformat(),
                ),
            )

        conn.commit()
        logger.info("Saved run %s to %s", run.run_id, db_path)
    finally:
        conn.close()


def _row_to_eval_run(row: tuple) -> EvalRun:
    """Convert a database row tuple to an EvalRun model."""
    (
        run_id, prompt_version, model, provider, timestamp,
        total_cases, accuracy, avg_composite_summary_score,
        p95_latency_ms, total_input_tokens, total_output_tokens,
        total_cost_aud, status, difficulty_breakdown_json, case_results_json,
    ) = row

    difficulty_breakdown = json.loads(difficulty_breakdown_json)
    case_results_raw = json.loads(case_results_json)

    case_results = [
        CaseResult(
            **{
                **cr,
                "timestamp": datetime.fromisoformat(cr["timestamp"]),
                "category_match": bool(cr["category_match"]),
            }
        )
        for cr in case_results_raw
    ]

    return EvalRun(
        run_id=run_id,
        prompt_version=prompt_version,
        model=model,
        provider=provider,
        timestamp=datetime.fromisoformat(timestamp),
        total_cases=total_cases,
        accuracy=accuracy,
        avg_composite_summary_score=avg_composite_summary_score,
        p95_latency_ms=p95_latency_ms,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cost_aud=total_cost_aud,
        status=status,
        difficulty_breakdown=difficulty_breakdown,
        case_results=case_results,
    )


def get_run(run_id: str, db_path: str) -> EvalRun | None:
    """Retrieve a single EvalRun by its run_id, or None if not found."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        cursor = conn.execute(
            "SELECT * FROM eval_runs WHERE run_id = ?",
            (run_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_eval_run(row)
    finally:
        conn.close()


def get_latest_runs(n: int, db_path: str) -> list[EvalRun]:
    """Return the n most recent EvalRuns ordered by timestamp descending."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        cursor = conn.execute(
            "SELECT * FROM eval_runs ORDER BY timestamp DESC LIMIT ?",
            (n,),
        )
        rows = cursor.fetchall()
        return [_row_to_eval_run(row) for row in rows]
    finally:
        conn.close()