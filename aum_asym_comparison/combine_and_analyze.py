#!/usr/bin/env python3
"""Combine the legacy and August 2026 noisy-AL CSV exports.

The script removes only the eight previously identified duplicate ordinary
MaxHerding baselines from the legacy CSV. Failed and intentionally repeated
runs are retained and distinguished with explicit provenance/variant columns.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


LEGACY_KEY_FIELDS = (
    "final_labeled_size",
    "noise",
    "al_method",
    "noise_method",
    "loss",
)

ADDED_FIELDS = (
    "record_id",
    "source_csv_row_number",
    "source_csv_label",
    "result_batch",
    "code_variant",
    "planned_final_labeled_size",
    "synthetic_noise_rate",
    "asymmetric_noise_rate",
    "aum_type",
    "bootstrap_soft_type",
    "noise_name",
    "al_method_name",
    "noise_method_name",
    "loss_name",
    "legacy_duplicate_resolution",
    "comparison_id",
)

DISPLAY_NOISE = {
    "clean": "Clean labels",
    "sym": "Symmetric noise",
    "asym": "Asymmetric noise",
    "cifar100n": "CIFAR-100N human noise",
}

DISPLAY_AL = {
    "coreset": "CoreSet",
    "margin": "Margin",
    "maxherding": "MaxHerding",
    "probcover": "ProbCover",
    "random": "Random",
    "typiclust": "TypiClust",
    "npc_probcover_crossval": "NPC-ProbCover (cross-validation)",
    "npc_probcover_knn": "NPC-ProbCover (kNN)",
    "npc_probcover_aum": "NPC-ProbCover (AUM)",
    "npc_probcover_oracle": "NPC-ProbCover (oracle)",
    "npc_maxherding_crossval": "NPC-MaxHerding (cross-validation)",
    "npc_maxherding_knn": "NPC-MaxHerding (kNN)",
    "npc_maxherding_aum": "NPC-MaxHerding (AUM)",
    "npc_maxherding_oracle": "NPC-MaxHerding (oracle)",
}

DISPLAY_NOISE_METHOD = {
    "none": "None",
    "aum": "AUM filtering",
    "crossval": "Cross-validation cleaning",
    "label_cleaning": "Direct label cleaning",
    "temporal": "Temporal cleaning",
}

DISPLAY_LOSS = {
    "cross_entropy": "Cross-entropy",
    "gce": "Generalized cross-entropy (GCE)",
    "bootstrap_hard": "Bootstrap hard",
    "bootstrap_soft": "Bootstrap soft",
}

BASE_AL_METHODS = {
    "coreset",
    "margin",
    "maxherding",
    "probcover",
    "random",
    "typiclust",
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AssertionError(f"Missing header in {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fieldnames: list[str] | tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def planned_budget(row: dict[str, str]) -> str:
    if row["final_labeled_size"]:
        return row["final_labeled_size"]
    match = re.search(r"(?:/|_)(400|800)(?:/|_|$)", row["source_path"])
    if not match:
        raise AssertionError(f"Cannot infer planned budget from {row['source_path']}")
    return match.group(1)


def is_aum_run(row: dict[str, str]) -> bool:
    return row["noise_method"] == "aum" or row["al_method"].endswith("_aum")


def classify_batch(row: dict[str, str], source_label: str) -> str:
    if source_label == "previous_csv":
        return "legacy_grid_0p5_20260729"
    if "rerun_changed_lowbudgetaum_bootstrapsoft" in row["source_path"]:
        return "netta_aum_bootstrapsoft_rerun_0p5_20260820"
    if "total_framework2_asym03" in row["source_path"]:
        return "asym_0p3_full_grid_20260820"
    raise AssertionError(f"Unknown new-result batch: {row['source_path']}")


def synthetic_noise_rate(row: dict[str, str], batch: str) -> str:
    if row["noise"] == "clean":
        return "0.0"
    if row["noise"] == "cifar100n":
        return ""
    if batch == "asym_0p3_full_grid_20260820":
        assert row["noise"] == "asym"
        return "0.3"
    return "0.5"


def enrich(
    row: dict[str, str],
    *,
    source_label: str,
    source_row: int,
    duplicate_resolution: str,
) -> dict[str, str]:
    result = dict(row)
    batch = classify_batch(row, source_label)
    rate = synthetic_noise_rate(row, batch)
    aum_type = ""
    if is_aum_run(row):
        aum_type = "basic" if source_label == "previous_csv" else "Netta"
    bootstrap_type = ""
    if row["loss"] == "bootstrap_soft":
        bootstrap_type = "detached" if source_label == "previous_csv" else "non_detached"
    budget = planned_budget(row)
    result.update(
        {
            "record_id": f"{'old' if source_label == 'previous_csv' else 'new'}_r{source_row:04d}",
            "source_csv_row_number": str(source_row),
            "source_csv_label": (
                "original_features_456_runs.csv"
                if source_label == "previous_csv"
                else "original_features_better_aum_and_asym.csv"
            ),
            "result_batch": batch,
            "code_variant": "legacy_pre_b185814" if source_label == "previous_csv" else "b185814",
            "planned_final_labeled_size": budget,
            "synthetic_noise_rate": rate,
            "asymmetric_noise_rate": rate if row["noise"] == "asym" else "",
            "aum_type": aum_type,
            "bootstrap_soft_type": bootstrap_type,
            "noise_name": DISPLAY_NOISE.get(row["noise"], row["noise"]),
            "al_method_name": DISPLAY_AL.get(row["al_method"], row["al_method"]),
            "noise_method_name": DISPLAY_NOISE_METHOD.get(row["noise_method"], row["noise_method"]),
            "loss_name": DISPLAY_LOSS.get(row["loss"], row["loss"]),
            "legacy_duplicate_resolution": duplicate_resolution,
            "comparison_id": "__".join(
                [
                    f"budget={budget}",
                    f"noise={row['noise']}",
                    f"rate={rate or 'observed'}",
                    f"al={row['al_method']}",
                    f"cleaning={row['noise_method']}",
                    f"loss={row['loss']}",
                    f"aum={aum_type or 'na'}",
                    f"bootstrap_soft={bootstrap_type or 'na'}",
                ]
            ),
        }
    )
    return result


def remove_known_legacy_duplicates(
    rows: list[dict[str, str]],
) -> tuple[list[tuple[int, dict[str, str], str]], list[tuple[int, dict[str, str]]]]:
    groups: dict[tuple[str, ...], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for source_row, row in enumerate(rows, start=2):
        groups[tuple(row[field] for field in LEGACY_KEY_FIELDS)].append((source_row, row))

    duplicate_groups = {key: group for key, group in groups.items() if len(group) > 1}
    assert len(duplicate_groups) == 8, duplicate_groups.keys()
    for key, group in duplicate_groups.items():
        assert len(group) == 2
        assert key[2:] == ("maxherding", "none", "cross_entropy")
        metric_fields = (
            "linear_train_acc",
            "linear_val_acc",
            "linear_test_acc",
            "knn_train_acc",
            "knn_val_acc",
            "knn_test_acc",
            "fixed_labels_total",
            "correctly_fixed_labels_total",
            "clean_corrupted_total",
            "wrong_to_wrong_total",
        )
        assert all(group[0][1][field] == group[1][1][field] for field in metric_fields)

    kept: list[tuple[int, dict[str, str], str]] = []
    removed: list[tuple[int, dict[str, str]]] = []
    seen: set[tuple[str, ...]] = set()
    for source_row, row in enumerate(rows, start=2):
        key = tuple(row[field] for field in LEGACY_KEY_FIELDS)
        if key in seen:
            assert key in duplicate_groups
            removed.append((source_row, row))
            continue
        seen.add(key)
        resolution = "kept_canonical_of_duplicate_pair" if key in duplicate_groups else "not_a_legacy_duplicate"
        kept.append((source_row, row, resolution))
    assert len(kept) == 448
    assert len(removed) == 8
    return kept, removed


def float_value(row: dict[str, str], field: str) -> float:
    value = row[field]
    if value == "":
        raise AssertionError(f"Missing {field} for completed row {row.get('record_id', row.get('run_key'))}")
    return float(value)


def comparison_group(row: dict[str, str]) -> str:
    if row["al_method"] in BASE_AL_METHODS and row["noise_method"] == "none" and row["loss"] == "cross_entropy":
        return "standard_ce"
    if row["al_method"] in BASE_AL_METHODS and row["noise_method"] in {"label_cleaning", "crossval", "temporal"}:
        return "direct_cleaning_stable_code"
    if row["al_method"] in BASE_AL_METHODS and row["noise_method"] == "none" and row["loss"] in {"gce", "bootstrap_hard"}:
        return "robust_loss_stable_code"
    if row["al_method"] in BASE_AL_METHODS and row["noise_method"] == "none" and row["loss"] == "bootstrap_soft":
        return "bootstrap_soft_noise_and_code_changed"
    if row["al_method"] in {"npc_probcover_crossval", "npc_probcover_knn", "npc_probcover_oracle"}:
        return "npc_probcover_stable_code"
    return "other"


def matched_asym_rows(combined: list[dict[str, str]]) -> list[dict[str, object]]:
    old = {
        (
            row["planned_final_labeled_size"],
            row["al_method"],
            row["noise_method"],
            row["loss"],
        ): row
        for row in combined
        if row["result_batch"] == "legacy_grid_0p5_20260729" and row["noise"] == "asym"
    }
    new_rows = [
        row
        for row in combined
        if row["result_batch"] == "asym_0p3_full_grid_20260820" and row["status"] == "completed"
    ]
    output: list[dict[str, object]] = []
    for new in new_rows:
        key = (
            new["planned_final_labeled_size"],
            new["al_method"],
            new["noise_method"],
            new["loss"],
        )
        previous = old[key]
        output.append(
            {
                "planned_final_labeled_size": new["planned_final_labeled_size"],
                "al_method": new["al_method"],
                "noise_method": new["noise_method"],
                "loss": new["loss"],
                "comparison_group": comparison_group(new),
                "old_asymmetric_noise_rate": "0.5",
                "new_asymmetric_noise_rate": "0.3",
                "old_aum_type": previous["aum_type"],
                "new_aum_type": new["aum_type"],
                "old_bootstrap_soft_type": previous["bootstrap_soft_type"],
                "new_bootstrap_soft_type": new["bootstrap_soft_type"],
                "old_knn_test_acc": previous["knn_test_acc"],
                "new_knn_test_acc": new["knn_test_acc"],
                "knn_test_acc_delta_points": float_value(new, "knn_test_acc") - float_value(previous, "knn_test_acc"),
                "old_linear_test_acc": previous["linear_test_acc"],
                "new_linear_test_acc": new["linear_test_acc"],
                "linear_test_acc_delta_points": float_value(new, "linear_test_acc") - float_value(previous, "linear_test_acc"),
                "old_record_id": previous["record_id"],
                "new_record_id": new["record_id"],
            }
        )
    assert len(output) == 84
    return output


def analysis_summary(
    combined: list[dict[str, str]], matched: list[dict[str, object]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(section: str, scope: str, metric: str, n: int, value: object, note: str = "") -> None:
        rows.append(
            {
                "section": section,
                "scope": scope,
                "metric": metric,
                "n": n,
                "value": value,
                "note": note,
            }
        )

    for batch in sorted({row["result_batch"] for row in combined}):
        batch_rows = [row for row in combined if row["result_batch"] == batch]
        counts = Counter(row["status"] for row in batch_rows)
        add("run_counts", batch, "all_rows", len(batch_rows), len(batch_rows))
        for status in sorted(counts):
            add("run_counts", batch, f"status={status}", counts[status], counts[status])

    aum = [row for row in combined if row["aum_type"]]
    for aum_type in ("basic", "Netta"):
        typed = [row for row in aum if row["aum_type"] == aum_type]
        counts = Counter(row["status"] for row in typed)
        add("aum_outcomes", f"aum_type={aum_type}", "all_rows", len(typed), len(typed))
        for status in sorted(counts):
            add("aum_outcomes", f"aum_type={aum_type}", f"status={status}", counts[status], counts[status])
    new_aum_failed = [row for row in aum if row["aum_type"] == "Netta" and row["status"] == "failed"]
    for reason, count in Counter(row["reason"] for row in new_aum_failed).most_common():
        add("aum_failure_reasons", "aum_type=Netta", reason, count, count)

    groups = sorted({str(row["comparison_group"]) for row in matched})
    for group in groups:
        group_rows = [row for row in matched if row["comparison_group"] == group]
        for budget in ("all", "400", "800"):
            scoped = group_rows if budget == "all" else [row for row in group_rows if row["planned_final_labeled_size"] == budget]
            if not scoped:
                continue
            for metric in ("knn_test_acc_delta_points", "linear_test_acc_delta_points"):
                values = [float(row[metric]) for row in scoped]
                add("asym_0p3_vs_0p5", f"{group};budget={budget}", f"mean_{metric}", len(values), mean(values))
                add("asym_0p3_vs_0p5", f"{group};budget={budget}", f"positive_{metric}", len(values), sum(value > 0 for value in values))

    # Compare each practical method with its same-rate CE baseline. These
    # comparisons address noise handling rather than the easier 0.3 setting.
    asym03_completed = [
        row
        for row in combined
        if row["result_batch"] == "asym_0p3_full_grid_20260820" and row["status"] == "completed"
    ]
    asym03 = {
        (
            row["planned_final_labeled_size"],
            row["al_method"],
            row["noise_method"],
            row["loss"],
        ): row
        for row in asym03_completed
    }

    for noise_method in ("label_cleaning", "crossval", "temporal"):
        for budget in ("all", "400", "800"):
            deltas_knn: list[float] = []
            deltas_linear: list[float] = []
            for planned in ("400", "800"):
                if budget != "all" and planned != budget:
                    continue
                for al_method in sorted(BASE_AL_METHODS):
                    current = asym03.get((planned, al_method, noise_method, "cross_entropy"))
                    baseline = asym03.get((planned, al_method, "none", "cross_entropy"))
                    if current is None or baseline is None:
                        continue
                    deltas_knn.append(float_value(current, "knn_test_acc") - float_value(baseline, "knn_test_acc"))
                    deltas_linear.append(float_value(current, "linear_test_acc") - float_value(baseline, "linear_test_acc"))
            if not deltas_knn:
                continue
            scope = f"method={noise_method};budget={budget}"
            add("asym_0p3_method_vs_ce", scope, "mean_knn_test_acc_delta_points", len(deltas_knn), mean(deltas_knn))
            add("asym_0p3_method_vs_ce", scope, "positive_knn_test_acc_delta", len(deltas_knn), sum(value > 0 for value in deltas_knn))
            add("asym_0p3_method_vs_ce", scope, "mean_linear_test_acc_delta_points", len(deltas_linear), mean(deltas_linear))

    for loss in ("gce", "bootstrap_hard", "bootstrap_soft"):
        for budget in ("all", "400", "800"):
            deltas: list[float] = []
            for planned in ("400", "800"):
                if budget != "all" and planned != budget:
                    continue
                for al_method in sorted(BASE_AL_METHODS):
                    current = asym03.get((planned, al_method, "none", loss))
                    baseline = asym03.get((planned, al_method, "none", "cross_entropy"))
                    if current is None or baseline is None:
                        continue
                    deltas.append(float_value(current, "linear_test_acc") - float_value(baseline, "linear_test_acc"))
            if not deltas:
                continue
            scope = f"loss={loss};budget={budget}"
            add("asym_0p3_loss_vs_ce", scope, "mean_linear_test_acc_delta_points", len(deltas), mean(deltas))
            add("asym_0p3_loss_vs_ce", scope, "positive_linear_test_acc_delta", len(deltas), sum(value > 0 for value in deltas))

    for detector in ("crossval", "knn", "oracle"):
        deltas_knn: list[float] = []
        deltas_linear: list[float] = []
        for planned in ("400", "800"):
            current = asym03.get((planned, f"npc_probcover_{detector}", "none", "cross_entropy"))
            baseline = asym03.get((planned, "probcover", "none", "cross_entropy"))
            if current is None or baseline is None:
                continue
            deltas_knn.append(float_value(current, "knn_test_acc") - float_value(baseline, "knn_test_acc"))
            deltas_linear.append(float_value(current, "linear_test_acc") - float_value(baseline, "linear_test_acc"))
        add("asym_0p3_npc_vs_standard", f"selector=probcover;detector={detector}", "mean_knn_test_acc_delta_points", len(deltas_knn), mean(deltas_knn))
        add("asym_0p3_npc_vs_standard", f"selector=probcover;detector={detector}", "mean_linear_test_acc_delta_points", len(deltas_linear), mean(deltas_linear))

    # Same-rate comparison isolates the bootstrap-soft implementation change.
    legacy = {
        (
            row["planned_final_labeled_size"],
            row["noise"],
            row["al_method"],
            row["noise_method"],
            row["loss"],
        ): row
        for row in combined
        if row["result_batch"] == "legacy_grid_0p5_20260729"
    }
    changed_bootstrap = [
        row
        for row in combined
        if row["result_batch"] == "netta_aum_bootstrapsoft_rerun_0p5_20260820"
        and row["status"] == "completed"
        and row["loss"] == "bootstrap_soft"
    ]
    assert len(changed_bootstrap) == 48
    for metric in ("knn_test_acc", "linear_test_acc"):
        deltas = []
        for current in changed_bootstrap:
            key = (
                current["planned_final_labeled_size"],
                current["noise"],
                current["al_method"],
                current["noise_method"],
                current["loss"],
            )
            previous = legacy[key]
            deltas.append(float_value(current, metric) - float_value(previous, metric))
        add("bootstrap_soft_same_rate", "non_detached_vs_detached", f"mean_{metric}_delta_points", len(deltas), mean(deltas))
        add("bootstrap_soft_same_rate", "non_detached_vs_detached", f"positive_{metric}_delta", len(deltas), sum(delta > 0 for delta in deltas))
        add("bootstrap_soft_same_rate", "non_detached_vs_detached", f"negative_{metric}_delta", len(deltas), sum(delta < 0 for delta in deltas))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-csv", type=Path, required=True)
    parser.add_argument("--new-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    old_fields, old_rows = read_csv(args.old_csv)
    new_fields, new_rows = read_csv(args.new_csv)
    assert old_fields == new_fields
    assert len(old_rows) == 456
    assert len(new_rows) == 224
    assert Counter(row["status"] for row in old_rows) == {"completed": 456}
    assert Counter(row["status"] for row in new_rows) == {"completed": 132, "failed": 92}

    kept_old, removed_old = remove_known_legacy_duplicates(old_rows)
    combined: list[dict[str, str]] = []
    for source_row, row, resolution in kept_old:
        combined.append(
            enrich(
                row,
                source_label="previous_csv",
                source_row=source_row,
                duplicate_resolution=resolution,
            )
        )
    for source_row, row in enumerate(new_rows, start=2):
        combined.append(
            enrich(
                row,
                source_label="new_csv",
                source_row=source_row,
                duplicate_resolution="not_from_legacy_duplicate_set",
            )
        )

    assert len(combined) == 672
    assert Counter(row["source_csv_label"] for row in combined) == {
        "original_features_456_runs.csv": 448,
        "original_features_better_aum_and_asym.csv": 224,
    }
    assert Counter(row["status"] for row in combined) == {"completed": 580, "failed": 92}
    assert Counter(row["planned_final_labeled_size"] for row in combined) == {"400": 336, "800": 336}
    assert sum(row["aum_type"] == "Netta" for row in combined) == 80
    assert all(row["status"] == "failed" for row in combined if row["aum_type"] == "Netta")
    completed = [row for row in combined if row["status"] == "completed"]
    for row in completed:
        for metric in ("knn_test_acc", "linear_test_acc"):
            float_value(row, metric)

    output_fields = list(ADDED_FIELDS) + old_fields
    write_csv(args.output_dir / "combined_runs_old_and_new.csv", output_fields, combined)

    removed_enriched = [
        enrich(
            row,
            source_label="previous_csv",
            source_row=source_row,
            duplicate_resolution="removed_duplicate_of_canonical_legacy_row",
        )
        for source_row, row in removed_old
    ]
    write_csv(args.output_dir / "removed_legacy_duplicates.csv", output_fields, removed_enriched)

    matched = matched_asym_rows(combined)
    matched_fields = list(matched[0].keys())
    write_csv(args.output_dir / "asym_03_vs_05_matched_completed.csv", matched_fields, matched)

    summary = analysis_summary(combined, matched)
    write_csv(
        args.output_dir / "analysis_summary.csv",
        ["section", "scope", "metric", "n", "value", "note"],
        summary,
    )

    print(f"combined_rows={len(combined)}")
    print(f"removed_legacy_duplicates={len(removed_enriched)}")
    print(f"completed_rows={sum(row['status'] == 'completed' for row in combined)}")
    print(f"failed_rows={sum(row['status'] == 'failed' for row in combined)}")
    print(f"matched_asym_completed_pairs={len(matched)}")


if __name__ == "__main__":
    main()
