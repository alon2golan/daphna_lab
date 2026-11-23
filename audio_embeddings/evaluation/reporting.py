"""Aggregate per-run audio AL scores without mixing experiment settings."""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from ..data.dcase import dcase_task_number


IDENTITY = ("dataset", "backbone", "adaptation", "method", "delta", "budget_unit", "nominal_budget")
NON_METRICS = {"seed", "fold", "nominal_budget", "nominal_query_target", "actual_queries",
               "budget_shortfall_unavailable_audio"}


def _csv_value(value):
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value


def _sort_budget(row):
    return (str(row.get("dataset", "")), str(row.get("backbone", "")),
            str(row.get("adaptation", "")), str(row.get("method", "")),
            float(row.get("delta") or 0), float(row.get("nominal_budget") or 0))


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    if not fields:
        return
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)
    _atomic_write(path, buffer.getvalue())


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _aggregate(rows: list[dict]) -> list[dict]:
    metric_fields = sorted({key for row in rows for key, value in row.items()
                            if isinstance(value, (int, float)) and key not in NON_METRICS
                            and key not in IDENTITY})
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(json.dumps(row.get(field), sort_keys=True) for field in IDENTITY)
        groups.setdefault(key, []).append(row)
    summary = []
    for values in groups.values():
        item = {key: values[0].get(key) for key in IDENTITY}
        item["runs"] = len(values)
        item["mean_actual_queries"] = float(np.mean([r["actual_queries"] for r in values if "actual_queries" in r]))
        for metric in metric_fields:
            observed = [float(r[metric]) for r in values if r.get(metric) is not None]
            if observed:
                item[f"{metric}_mean"] = float(np.mean(observed))
                item[f"{metric}_std"] = float(np.std(observed, ddof=1)) if len(observed) > 1 else 0.0
        summary.append(item)
    return sorted(summary, key=_sort_budget)


def _summary_markdown(summary: list[dict]) -> str:
    lines = ["# Audio active learning score summary", "",
             "Mean +/- standard deviation across available seeds/folds. Settings remain grouped by dataset, backbone, adaptation, selector, and budget.", "",
             "| Dataset | Backbone | Mode | Method | Budget | Mean queries | Runs | Metrics |",
             "|---|---|---|---|---:|---:|---:|---|"]
    metric_fields = sorted({key[:-5] for row in summary for key in row if key.endswith("_mean")
                            and key not in {"mean_actual_queries"}})
    for row in summary:
        metrics = "; ".join(f"{metric}: {row[metric + '_mean']:.4f} +/- {row.get(metric + '_std', 0):.4f}"
                             for metric in metric_fields if metric + "_mean" in row)
        budget = f"{row.get('nominal_budget')} {row.get('budget_unit')}"
        if row.get("delta") is not None:
            budget += f" (delta={row['delta']})"
        lines.append(f"| {row.get('dataset')} | {row.get('backbone')} | {row.get('adaptation')} | {row.get('method')} | {budget} | {row.get('mean_actual_queries', 0):.1f} | {row['runs']} | {metrics} |")
    return "\n".join(lines) + "\n"


def _plot_summaries(run_dir: Path, summary: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    wanted = {"esc50": ("mAP", "accuracy", "macro_f1"),
              "as20k": ("mAP", "micro_f1", "macro_f1"),
              "dcase": ("mAP", "macro_f1", "segment_er", "segment_f_score")}
    task_metric = {"dcase2016": ("accuracy", "macro_f1"),
                   "dcase": ("accuracy", "macro_f1"),
                   "dcase2016_1": ("accuracy", "macro_f1"),
                   "dcase20161": ("accuracy", "macro_f1"),
                   "dcase2016_4": wanted["dcase"],
                   "dcase20164": wanted["dcase"]}
    combinations = sorted({(row.get("dataset"), row.get("backbone"), row.get("adaptation")) for row in summary})
    for dataset, backbone, mode in combinations:
        dcase_task = dcase_task_number(str(dataset))
        metrics = (("accuracy", "macro_f1") if dcase_task == 1 else
                   wanted["dcase"] if dcase_task == 4 else
                   wanted.get(str(dataset).lower(), ("mAP",)))
        subset = [row for row in summary if (row.get("dataset"), row.get("backbone"), row.get("adaptation")) == (dataset, backbone, mode)]
        methods = sorted({(row.get("method"), row.get("delta")) for row in subset})
        for metric in metrics:
            if not any(f"{metric}_mean" in row for row in subset):
                continue
            fig, (ax, low) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
            max_budget = max(float(row.get("nominal_budget") or 0) for row in subset)
            low_cutoff = max(1.0, max_budget * 0.25)
            for method, delta in methods:
                points = sorted((r for r in subset if r.get("method") == method and r.get("delta") == delta
                                 and f"{metric}_mean" in r), key=lambda r: float(r.get("nominal_budget") or 0))
                if not points:
                    continue
                label = str(method) + (f" delta={delta}" if delta is not None else "")
                x = [float(point.get("mean_actual_queries", point.get("nominal_query_target", 0))) for point in points]
                y = [float(point[f"{metric}_mean"]) for point in points]
                e = [float(point.get(f"{metric}_std", 0)) for point in points]
                ax.errorbar(x, y, yerr=e, marker="o", capsize=2, label=label)
                low_points = [(xx, yy, ee) for xx, yy, ee, p in zip(x, y, e, points)
                              if float(p.get("nominal_budget") or 0) <= low_cutoff]
                if low_points:
                    lx, ly, le = zip(*low_points)
                    low.errorbar(lx, ly, yerr=le, marker="o", capsize=2, label=label)
            for axis, title in ((ax, "All configured budgets"), (low, "Low-budget view")):
                axis.set_xlabel("Actual queried examples (mean)")
                axis.set_title(title)
                axis.grid(alpha=0.25)
            ax.set_ylabel(metric)
            ax.legend(fontsize="small")
            fig.suptitle(f"{dataset} / {backbone} / {mode} / {metric}")
            fig.tight_layout()
            safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{dataset}-{backbone}-{mode}-{metric}")
            fig.savefig(run_dir / f"{safe}.png", dpi=160)
            plt.close(fig)


def write_scores(run_dir: Path, rows: list[dict]) -> None:
    """Write raw per-run scores and setting-aware aggregates/plots."""
    if not rows:
        return
    rows = sorted(rows, key=_sort_budget)
    _write_csv(run_dir / "scores.csv", rows)
    summary = _aggregate(rows)
    _write_csv(run_dir / "score_summary.csv", summary)
    _atomic_write(run_dir / "score_summary.md", _summary_markdown(summary))
    _plot_summaries(run_dir, summary)
