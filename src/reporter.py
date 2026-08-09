"""HTML report generator — renders a self-contained diff report from eval run data."""

from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.models import CaseResult, EvalRun, GoldenCase, GoldenDataset, RunDiff

logger = logging.getLogger(__name__)

ReportRow = dict[str, str | float | None]

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_CATEGORIES = ["billing", "technical", "account", "general"]
_DIFFICULTY_ORDER = ["easy", "medium", "hard", "edge"]


def generate_report(
    run: EvalRun,
    diff: RunDiff,
    drift_warning: bool,
    drift_summary: str,
    recent_runs: list[EvalRun],
    output_dir: str,
    dataset: GoldenDataset,
) -> str:
    """Render the HTML report and write it to output_dir; return the file path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prev_run = _get_previous_run(run, recent_runs)

    regression_table = build_regression_table_data(run, diff, dataset)
    improvement_table = build_improvement_table_data(run, diff, dataset)

    accuracy_trend_svg = _sparkline_svg(
        [r.accuracy * 100 for r in reversed(recent_runs)],
        label="Accuracy %",
        fmt="{:.1f}%",
    )
    cost_trend_svg = _sparkline_svg(
        [r.total_cost_aud for r in reversed(recent_runs)],
        label="Cost AUD",
        fmt="${:.4f}",
    )

    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=True)
    template = env.get_template("report.html.j2")

    html = template.render(
        run=run,
        diff=diff,
        prev_run=prev_run,
        drift_warning=drift_warning,
        drift_summary=drift_summary,
        categories=_CATEGORIES,
        difficulty_order=_DIFFICULTY_ORDER,
        regression_table=regression_table,
        improvement_table=improvement_table,
        accuracy_trend_svg=accuracy_trend_svg,
        cost_trend_svg=cost_trend_svg,
    )

    report_path = out / f"report_{run.run_id[:8]}.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", report_path)
    return str(report_path)


def build_regression_table_data(
    run: EvalRun, diff: RunDiff, dataset: GoldenDataset
) -> list[ReportRow]:
    """Build row data for the regression cases table from the run diff."""
    if not diff.regression_cases:
        return []
    case_map = {c.id: c for c in dataset.cases}
    result_map = {cr.case_id: cr for cr in run.case_results}
    return _build_case_table_rows(diff.regression_cases, case_map, result_map)


def build_improvement_table_data(
    run: EvalRun, diff: RunDiff, dataset: GoldenDataset
) -> list[ReportRow]:
    """Build row data for the improvement cases table from the run diff."""
    if not diff.improvement_cases:
        return []
    case_map = {c.id: c for c in dataset.cases}
    result_map = {cr.case_id: cr for cr in run.case_results}
    return _build_case_table_rows(diff.improvement_cases, case_map, result_map)


def _build_case_table_rows(
    case_ids: list[str],
    case_map: dict[str, GoldenCase],
    result_map: dict[str, CaseResult],
) -> list[ReportRow]:
    """Map a list of case IDs into table row dicts."""
    rows: list[ReportRow] = []
    for cid in case_ids:
        golden = case_map.get(cid)
        cr = result_map.get(cid)
        if golden is None or cr is None:
            continue
        email_text = golden.input
        if len(email_text) > 80:
            email_text = email_text[:77] + "..."
        rows.append({
            "case_id": cid,
            "email": email_text,
            "expected": cr.expected_category,
            "predicted": cr.predicted_category,
            "score": cr.composite_summary_score,
            "reason": cr.judge_reason,
            "adversarial_target": golden.adversarial_target,
        })
    return rows


def _get_previous_run(current: EvalRun, recent_runs: list[EvalRun]) -> EvalRun | None:
    """Return the most recent run that is not the current run."""
    for r in recent_runs:
        if r.run_id != current.run_id:
            return r
    return None


def _sparkline_svg(
    values: list[float],
    label: str = "",
    fmt: str = "{:.2f}",
    width: int = 480,
    height: int = 80,
) -> str:
    """Generate an inline SVG sparkline from a list of numeric values."""
    if not values:
        return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"><text x="10" y="45" fill="#94a3b8" font-size="12">No data</text></svg>'

    pad_x, pad_y = 10, 14
    plot_w = width - 2 * pad_x
    plot_h = height - 2 * pad_y

    vmin = min(values)
    vmax = max(values)
    vrange = vmax - vmin if vmax != vmin else 1.0

    points: list[str] = []
    for i, v in enumerate(values):
        x = pad_x + (i / max(len(values) - 1, 1)) * plot_w
        y = pad_y + plot_h - ((v - vmin) / vrange) * plot_h
        points.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(points)
    last_x = pad_x + plot_w
    last_y = pad_y + plot_h - ((values[-1] - vmin) / vrange) * plot_h

    svg_parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{width}" height="{height}" fill="#1e293b" rx="4"/>',
        f'<polyline points="{polyline}" fill="none" stroke="#38bdf8" stroke-width="1.5"/>',
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="3" fill="#38bdf8"/>',
        f'<text x="{pad_x}" y="11" fill="#94a3b8" font-size="10">{label}</text>',
        f'<text x="{last_x:.1f}" y="{last_y - 6:.1f}" fill="#e2e8f0" font-size="10" text-anchor="end">{fmt.format(values[-1])}</text>',
        f'<text x="{pad_x}" y="{height - 2}" fill="#94a3b8" font-size="9">{fmt.format(vmin)}</text>',
        f'<text x="{width - pad_x}" y="{height - 2}" fill="#94a3b8" font-size="9" text-anchor="end">{fmt.format(vmax)}</text>',
        "</svg>",
    ]
    return "\n".join(svg_parts)
