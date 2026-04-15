"""
HTML report generator: produce a fully self-contained HTML file from a
results DataFrame.  No internet connection is required to view the output
because Plotly JS is embedded inline.

Usage
-----
    from reporting.html_export import generate_report
    from reporting.scanner import load_all_results

    df = load_all_results()
    html = generate_report(df, title="CIFAR-100 AL Comparison")
    open("report.html", "w").write(html)
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[1]

_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

_CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 2rem 3rem;
  color: #333;
  background: #fff;
}
h1 { color: #1a1a2e; margin-bottom: 0.25rem; }
h2 { color: #2c3e50; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }
.meta { color: #777; font-size: 0.88rem; margin-bottom: 2rem; }
table.summary {
  border-collapse: collapse;
  width: 100%;
  margin: 1.2rem 0 2rem;
  font-size: 0.9rem;
}
table.summary th {
  background: #f0f4f8;
  font-weight: 600;
  text-align: left;
  padding: 0.55rem 0.75rem;
  border: 1px solid #d0d7de;
}
table.summary td {
  padding: 0.45rem 0.75rem;
  border: 1px solid #d0d7de;
}
table.summary tr:nth-child(even) td { background: #f8fafc; }
"""


# ── Figure builders ───────────────────────────────────────────────────────────


def _build_comparison_figure(agg_list: list) -> go.Figure:
    """Mean ± std band line chart for a list of aggregated config dicts."""
    fig = go.Figure()
    for i, agg in enumerate(agg_list):
        if agg["agg"].empty:
            continue
        color = _COLORS[i % len(_COLORS)]
        agg_df = agg["agg"].reset_index(drop=True)
        x = list(agg_df["episode"])
        mean = list(agg_df["test_acc_mean"])
        std = list(agg_df["test_acc_std"])
        label = f"{agg['sweep_name']} / {agg['config_label']}"

        upper = [m + s for m, s in zip(mean, std)]
        lower = [m - s for m, s in zip(mean, std)]

        # Shaded std band
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor=color,
                opacity=0.15,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Mean line
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean,
                mode="lines+markers",
                name=f"{label} (n={agg['n_seeds']})",
                line=dict(color=color, width=2.5),
                marker=dict(size=7),
            )
        )

    fig.update_layout(
        title="Test Accuracy per Episode (mean ± std across seeds)",
        xaxis_title="Episode",
        yaxis_title="Test Accuracy (%)",
        template="plotly_white",
        height=520,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        hovermode="x unified",
    )
    return fig


# ── HTML helpers ──────────────────────────────────────────────────────────────


def _summary_table_html(agg_list: list) -> str:
    rows_html = ""
    for agg in agg_list:
        mean_str = (
            f"{agg['final_test_acc_mean']:.2f}%"
            if agg["final_test_acc_mean"] is not None
            else "—"
        )
        std_str = (
            f"± {agg['final_test_acc_std']:.2f}%"
            if agg["final_test_acc_std"] is not None
            else "—"
        )
        rows_html += (
            f"<tr>"
            f"<td>{agg['sweep_name']}</td>"
            f"<td>{agg['config_label']}</td>"
            f"<td style='text-align:center'>{agg['n_seeds']}</td>"
            f"<td style='text-align:right'>{mean_str}</td>"
            f"<td style='text-align:right'>{std_str}</td>"
            f"</tr>\n"
        )
    return (
        '<table class="summary">\n'
        "  <thead><tr>"
        "<th>Sweep</th><th>Config</th><th>Seeds</th>"
        "<th>Final Test Acc (mean)</th><th>Std</th>"
        "</tr></thead>\n"
        f"  <tbody>\n{rows_html}  </tbody>\n"
        "</table>\n"
    )


# ── Public API ────────────────────────────────────────────────────────────────


def generate_report(
    df: pd.DataFrame,
    title: str = "AL Experiment Report",
) -> str:
    """
    Return a fully self-contained HTML string.

    Parameters
    ----------
    df : DataFrame returned by load_all_results() (may be pre-filtered).
    title : Title shown in the page <h1> and <title>.
    """
    if df.empty:
        return (
            "<!DOCTYPE html><html><head><title>No data</title></head>"
            "<body><h1>No data found</h1></body></html>"
        )

    # Import here to avoid circular dependency at module level
    from reporting.scanner import aggregate_config_episodes  # noqa: PLC0415

    agg_list = [
        aggregate_config_episodes(g)
        for _, g in df.groupby(["sweep_id", "config_key"], sort=False)
        if len(g) > 0
    ]
    agg_list = [a for a in agg_list if not a["agg"].empty]

    fig = _build_comparison_figure(agg_list)
    # include_plotlyjs=True embeds the full Plotly bundle — no network needed.
    chart_html = fig.to_html(include_plotlyjs=True, full_html=False)

    table_html = _summary_table_html(agg_list)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_runs = len(df)
    n_sweeps = df["sweep_id"].nunique()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>{title}</h1>
  <p class="meta">
    Generated: {generated_at} &nbsp;|&nbsp;
    {n_runs} run(s) &nbsp;|&nbsp; {n_sweeps} sweep(s)
  </p>

  <h2>Comparison Chart</h2>
  {chart_html}

  <h2>Final Accuracy Summary</h2>
  {table_html}
</body>
</html>
"""
