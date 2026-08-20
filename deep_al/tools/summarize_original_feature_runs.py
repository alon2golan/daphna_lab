#!/usr/bin/env python3
"""Combine original-feature experiment JSONs into one CSV.

Each input JSON may be in a different directory. Pass the full paths as
positional arguments.

Example:
    python summarize_original_feature_runs.py \
        /cs/labs/daphna/tomn/projects/daphna_lab/grid_outputs/total_framework2_seed42/aggregate_results_20260729_185638.json \
        /cs/labs/daphna/tomn/projects/daphna_lab/grid_outputs/total_framework2_seed42/aggregate_results_20260729_185653.json \
        /cs/labs/daphna/tomn/projects/daphna_lab/grid_outputs/npc_maxherding_seed42/aggregate_results_20260729_184900.json \
        /cs/labs/daphna/tomn/projects/daphna_lab/grid_outputs/npc_maxherding_seed42/aggregate_results_20260729_184909.json \
        --output /cs/labs/daphna/tomn/projects/daphna_lab/grid_outputs/original_features_456_runs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "2.0-cleaning-counters-fixed"

COLUMNS = [
    "experiment_family",
    "source_file",
    "source_path",
    "run_key",
    "status",
    "reason",
    "noise",
    "al_method",
    "noise_method",
    "loss",
    "method_variant",
    "seed",
    "features_tag",
    "initial_size",
    "budget_per_round",
    "max_iter",
    "final_labeled_size",
    "linear_train_acc",
    "linear_val_acc",
    "linear_test_acc",
    "knn_train_acc",
    "knn_val_acc",
    "knn_test_acc",
    "runtime_sec",
    "fixed_labels_total",
    "correctly_fixed_labels_total",
    "clean_corrupted_total",
    "wrong_to_wrong_total",
]


def get_nested(obj: Any, *keys: str, default: Any = None) -> Any:
    current = obj
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def parse_run_key(run_key: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in run_key.split("__"):
        if "=" in part:
            key, value = part.split("=", 1)
            result[key] = value
    return result


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def infer_family(data: dict[str, Any]) -> str:
    methods = {
        parse_run_key(run_key).get("al", "")
        for run_key in data
    }
    npc_maxherding_methods = {
        "maxherding",
        "npc_maxherding_crossval",
        "npc_maxherding_knn",
        "npc_maxherding_aum",
        "npc_maxherding_oracle",
    }
    if methods and methods.issubset(npc_maxherding_methods):
        return "npc_maxherding"
    return "many_cleaning_methods"


def infer_final_size(
    pipeline: dict[str, Any],
    summary: dict[str, Any],
) -> int | str:
    for key in (
        "final_lSet_len",
        "final_lset_size",
        "final_labeled_size",
        "final_labeled_set_size",
    ):
        value = summary.get(key)
        if isinstance(value, int):
            return value

    initial = get_nested(pipeline, "run_args", "initial_size")
    budget = get_nested(pipeline, "active_learning", "budget")
    max_iter = get_nested(pipeline, "active_learning", "max_iter")

    if all(isinstance(value, int) for value in (initial, budget, max_iter)):
        return initial + budget * max_iter
    return ""


def extract_runtime(summary: dict[str, Any]) -> float | str:
    for key in (
        "total_runtime_sec",
        "runtime_sec",
        "total_time_sec",
        "total_runtime",
    ):
        value = number(summary.get(key))
        if value is not None:
            return value

    episode_training = summary.get("episode_training_times")
    episode_al = summary.get("episode_al_times")
    if isinstance(episode_training, list) and isinstance(episode_al, list):
        values = [
            float(value)
            for value in episode_training + episode_al
            if number(value) is not None
        ]
        if values:
            return sum(values)

    return ""


def extract_cleaning_totals(
    entry: dict[str, Any],
    summary: dict[str, Any],
    status: str,
) -> dict[str, int | str]:
    """Aggregate cleaning counters across every AL episode."""

    result_keys = (
        "fixed",
        "correctly_fixed",
        "clean_corrupted",
        "wrong_to_wrong",
    )

    # Compact/previously aggregated JSON formats.
    for candidate in (
        summary.get("noise_cleaning_totals"),
        entry.get("noise_cleaning_totals"),
        get_nested(entry, "results", "noise_cleaning_totals"),
    ):
        if isinstance(candidate, dict):
            return {
                key: int(candidate.get(key, 0) or 0)
                for key in result_keys
            }

    # Full debug JSON format used by the July 29 grids:
    # results.summary.noise_method_results[*].result
    episode_results = summary.get("noise_method_results")
    if isinstance(episode_results, list):
        totals = {key: 0 for key in result_keys}

        for episode in episode_results:
            if not isinstance(episode, dict):
                continue

            result = episode.get("result", episode)
            if not isinstance(result, dict):
                continue

            if (
                result.get("status") == "skipped"
                or result.get("method") == "none"
            ):
                continue

            stats = result.get("stats")
            if not isinstance(stats, dict):
                stats = result

            for key in result_keys:
                try:
                    totals[key] += int(stats.get(key, 0) or 0)
                except (TypeError, ValueError):
                    pass

        return totals

    # A completed run without cleaning records means no relabeling was done.
    if status == "completed":
        return {key: 0 for key in result_keys}

    return {key: "" for key in result_keys}


def method_variant(
    al_method: str,
    noise_method: str,
    loss: str,
) -> str:
    if al_method.startswith("npc_"):
        return al_method.removeprefix("npc_")
    if loss and loss != "cross_entropy":
        return loss
    return noise_method


def original_features_tag(pipeline: dict[str, Any]) -> str:
    tag = get_nested(pipeline, "run_args", "features_tag")
    if tag in (None, ""):
        tag = pipeline.get("features_tag")

    normalized = "" if tag is None else str(tag).strip()
    if normalized.lower() not in {
        "",
        "none",
        "null",
        "default",
        "original",
    }:
        raise ValueError(
            f"Non-original features detected: features_tag={normalized!r}"
        )
    return "original/default"


def rows_from_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")

    family = infer_family(data)
    rows: list[dict[str, Any]] = []

    for run_key, entry in data.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: invalid entry for {run_key}")

        parsed = parse_run_key(run_key)
        pipeline = entry.get("pipeline", {})
        results = entry.get("results", {})
        summary = (
            results.get("summary", {})
            if isinstance(results, dict)
            else {}
        )

        if not isinstance(pipeline, dict):
            pipeline = {}
        if not isinstance(summary, dict):
            summary = {}

        status = (
            str(results.get("status", "missing"))
            if isinstance(results, dict)
            else "missing"
        )

        al_method = parsed.get(
            "al",
            str(get_nested(
                pipeline,
                "active_learning",
                "method",
                default="",
            )),
        )
        noise_method = parsed.get(
            "nm",
            str(get_nested(
                pipeline,
                "noise_handling",
                "method",
                default="",
            )),
        )
        loss = parsed.get(
            "loss",
            str(get_nested(
                pipeline,
                "run_args",
                "loss_fun",
                default="cross_entropy",
            )),
        )
        noise = parsed.get(
            "noise",
            str(get_nested(
                pipeline,
                "noise",
                "type",
                default="",
            )),
        )

        cleaning = extract_cleaning_totals(entry, summary, status)
        knn = summary.get("final_knn_eval", {})
        if not isinstance(knn, dict):
            knn = {}

        rows.append({
            "experiment_family": family,
            "source_file": path.name,
            "source_path": str(path),
            "run_key": run_key,
            "status": status,
            "reason": (
                results.get("reason", "")
                if isinstance(results, dict)
                else ""
            ),
            "noise": noise,
            "al_method": al_method,
            "noise_method": noise_method,
            "loss": loss,
            "method_variant": method_variant(
                al_method,
                noise_method,
                loss,
            ),
            "seed": parsed.get(
                "seed",
                get_nested(
                    pipeline,
                    "run_args",
                    "seed",
                    default="",
                ),
            ),
            "features_tag": original_features_tag(pipeline),
            "initial_size": get_nested(
                pipeline,
                "run_args",
                "initial_size",
                default="",
            ),
            "budget_per_round": get_nested(
                pipeline,
                "active_learning",
                "budget",
                default="",
            ),
            "max_iter": get_nested(
                pipeline,
                "active_learning",
                "max_iter",
                default="",
            ),
            "final_labeled_size": infer_final_size(
                pipeline,
                summary,
            ),
            "linear_train_acc": summary.get(
                "final_train_acc",
                "",
            ),
            "linear_val_acc": summary.get(
                "final_val_acc",
                "",
            ),
            "linear_test_acc": summary.get(
                "final_test_acc",
                "",
            ),
            "knn_train_acc": knn.get("train_acc", ""),
            "knn_val_acc": knn.get("val_acc", ""),
            "knn_test_acc": knn.get("test_acc", ""),
            "runtime_sec": extract_runtime(summary),
            "fixed_labels_total": cleaning["fixed"],
            "correctly_fixed_labels_total": cleaning[
                "correctly_fixed"
            ],
            "clean_corrupted_total": cleaning[
                "clean_corrupted"
            ],
            "wrong_to_wrong_total": cleaning[
                "wrong_to_wrong"
            ],
        })

    return rows


def sorting_key(row: dict[str, Any]) -> tuple[Any, ...]:
    noise_order = {
        "clean": 0,
        "sym": 1,
        "asym": 2,
        "cifar100n": 3,
    }
    final_size = row["final_labeled_size"]
    if not isinstance(final_size, int):
        final_size = 10**9

    return (
        row["experiment_family"],
        final_size,
        noise_order.get(str(row["noise"]), 99),
        row["al_method"],
        row["noise_method"],
        row["loss"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_paths",
        nargs="+",
        type=Path,
        help="Full paths to aggregate-result JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("original_features_all_runs.csv"),
    )
    args = parser.parse_args()

    print(f"Script version: {SCRIPT_VERSION}")

    all_rows: list[dict[str, Any]] = []
    for raw_path in args.json_paths:
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            parser.error(f"File does not exist: {path}")

        file_rows = rows_from_json(path)
        all_rows.extend(file_rows)

        completed = sum(
            row["status"] == "completed"
            for row in file_rows
        )
        print(
            f"{path}: {len(file_rows)} runs, "
            f"{completed} completed"
        )

    all_rows.sort(key=sorting_key)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    nonzero_cleaning_rows = sum(
        isinstance(row["fixed_labels_total"], int)
        and row["fixed_labels_total"] > 0
        for row in all_rows
    )
    total_label_changes = sum(
        row["fixed_labels_total"]
        for row in all_rows
        if isinstance(row["fixed_labels_total"], int)
    )

    direct_cleaning_candidates = [
        row
        for row in all_rows
        if row["status"] == "completed"
        and row["noise_method"] in {
            "label_cleaning",
            "temporal",
            "aum",
        }
        and isinstance(row["max_iter"], int)
        and row["max_iter"] >= 4
    ]

    print()
    print(f"Wrote {len(all_rows)} rows to: {output}")
    print(
        "Rows with at least one label change: "
        f"{nonzero_cleaning_rows}"
    )
    print(
        "Total label-change operations across the table: "
        f"{total_label_changes}"
    )

    if direct_cleaning_candidates and nonzero_cleaning_rows == 0:
        raise RuntimeError(
            "Cleaning-capable runs were found, but all cleaning counters "
            "are zero. The input JSON format was not parsed as expected."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
