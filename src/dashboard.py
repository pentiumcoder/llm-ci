"""Streamlit dashboard — historical eval run trends and drill-down."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.config import Settings
from src.storage import get_latest_runs

logger = logging.getLogger(__name__)

_STATUS_COLOURS = {"PASS": "#22c55e", "WARN": "#f59e0b", "FAIL": "#ef4444"}
_MAX_RUNS = 500


def _load_runs(db_path: str) -> list[dict]:
    """Load all eval runs from SQLite and return as a list of dicts."""
    runs = get_latest_runs(_MAX_RUNS, db_path)
    return [r.model_dump(mode="json") for r in reversed(runs)]


def _build_df(records: list[dict]) -> pd.DataFrame:
    """Convert run records into a filtered DataFrame."""
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    return df


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar provider and date range filters to the DataFrame."""
    if df.empty:
        return df

    providers = sorted(df["provider"].unique().tolist())
    selected = st.sidebar.multiselect("Provider", providers, default=providers)
    df = df[df["provider"].isin(selected)]

    if not df.empty:
        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()
        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if len(date_range) == 2:
            start, end = date_range
            df = df[
                (df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)
            ]

    return df


def _render_summary_cards(df: pd.DataFrame) -> None:
    """Render the four summary metric cards at the top of the page."""
    cols = st.columns(4)

    if df.empty:
        for col in cols:
            col.metric("—", "No data")
        return

    latest = df.iloc[-1]
    cols[0].metric("Latest Accuracy", f"{latest['accuracy'] * 100:.1f}%")
    cols[1].metric("Latest Cost (AUD)", f"${latest['total_cost_aud']:.4f}")
    cols[2].metric("Total Runs", str(len(df)))

    fail_rows = df[df["status"] == "FAIL"]
    if not fail_rows.empty:
        last_fail = fail_rows.iloc[-1]["timestamp"]
        cols[3].metric("Last FAIL", last_fail.strftime("%Y-%m-%d %H:%M"))
    else:
        cols[3].metric("Last FAIL", "None")


def _render_accuracy_chart(df: pd.DataFrame) -> None:
    """Render the accuracy line chart coloured by status."""
    if df.empty:
        return

    colour_map = {s: _STATUS_COLOURS.get(s, "#94a3b8") for s in df["status"].unique()}

    for status, colour in colour_map.items():
        subset = df[df["status"] == status]
        st.line_chart(
            subset.set_index("timestamp")[["accuracy"]].rename(
                columns={"accuracy": f"Accuracy ({status})"}
            ),
            color=colour,
            height=300,
        )


def _render_cost_chart(df: pd.DataFrame) -> None:
    """Render the total_cost_aud line chart."""
    if df.empty:
        return
    st.line_chart(
        df.set_index("timestamp")[["total_cost_aud"]],
        color="#38bdf8",
        height=250,
    )


def _render_composite_chart(df: pd.DataFrame) -> None:
    """Render the avg_composite_summary_score line chart."""
    if df.empty:
        return
    st.line_chart(
        df.set_index("timestamp")[["avg_composite_summary_score"]],
        color="#a78bfa",
        height=250,
    )


def _render_runs_table(df: pd.DataFrame) -> None:
    """Render the last 20 runs table with click-to-expand case breakdown."""
    if df.empty:
        st.info("No runs recorded yet.")
        return

    display_cols = [
        "run_id", "provider", "model", "prompt_version",
        "accuracy", "total_cost_aud", "status", "timestamp",
    ]
    table_df = df[display_cols].copy()
    table_df["run_id"] = table_df["run_id"].str[:8]
    table_df["accuracy"] = table_df["accuracy"].apply(lambda x: f"{x * 100:.1f}%")
    table_df["total_cost_aud"] = table_df["total_cost_aud"].apply(lambda x: f"${x:.4f}")
    table_df["timestamp"] = table_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    table_df = table_df.tail(20).iloc[::-1]

    st.dataframe(table_df, use_container_width=True, hide_index=True)

    full_ids = df["run_id"].tolist()
    short_ids = [rid[:8] for rid in full_ids]
    selected = st.selectbox("Select run to expand", short_ids, index=0)
    if selected:
        idx = short_ids.index(selected)
        full_id = full_ids[idx]
        run_record = df[df["run_id"] == full_id].iloc[0].to_dict()
        _render_difficulty_heatmap(run_record)


def _render_difficulty_heatmap(run_record: dict) -> None:
    """Render a colour-styled difficulty breakdown table for the selected run."""
    breakdown = run_record.get("difficulty_breakdown", {})
    if not breakdown:
        st.caption("No difficulty breakdown available for this run.")
        return

    st.subheader(f"Difficulty Breakdown — {run_record['run_id'][:8]}")

    rows = []
    for difficulty, cats in sorted(breakdown.items()):
        for cat, acc in sorted(cats.items()):
            rows.append({"difficulty": difficulty, "category": cat, "accuracy": acc})

    if not rows:
        st.caption("Empty difficulty breakdown.")
        return

    heat_df = pd.DataFrame(rows)
    pivot = heat_df.pivot(index="difficulty", columns="category", values="accuracy")
    pivot = pivot.reindex(index=["easy", "medium", "hard", "edge"], errors="ignore")

    def _color_cell(val: float) -> str:
        """Return a background colour string based on accuracy thresholds."""
        if val >= 0.85:
            return "background-color: #22c55e33"
        if val >= 0.70:
            return "background-color: #f59e0b33"
        return "background-color: #ef444433"

    styled = pivot.style.applymap(_color_cell).format("{:.0%}")
    st.dataframe(styled, use_container_width=True)


def main() -> None:
    """Entry point for the Streamlit dashboard app."""
    st.set_page_config(page_title="LLM-CI — Eval History", layout="wide")
    st.title("LLM-CI — Eval History")

    settings = Settings()
    records = _load_runs(settings.db_path)
    df = _build_df(records)
    df = _apply_filters(df)

    _render_summary_cards(df)

    st.subheader("Accuracy Trend")
    _render_accuracy_chart(df)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Cost Trend (AUD)")
        _render_cost_chart(df)
    with col_b:
        st.subheader("Composite Summary Score Trend")
        _render_composite_chart(df)

    st.subheader("Run History")
    _render_runs_table(df)


if __name__ == "__main__":
    main()
