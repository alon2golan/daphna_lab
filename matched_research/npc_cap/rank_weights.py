"""Keep the official NPC labels and change one six-hop vote weight."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np

from filter_swap import _midranks
from replay import BANK_HASHES, BUDGETS, CASES, file_sha256, path_matrix, read, six_edge_counts


def class_weights(labels, aum):
    weights = np.empty(len(labels), dtype=np.float64)
    for c in np.unique(labels):
        rows = np.flatnonzero(labels == c)
        weights[rows] = 0.5 + (_midranks(aum[rows]) + 0.5) / len(rows)
    return weights


def predict(paths, labels):
    return (paths @ np.eye(100, dtype=paths.dtype)[labels]).argmax(1)


def score(study, frozen_graph, task, output):
    condition, seed = CASES[task]
    case = f"{condition}_s{seed}"
    proof = read(frozen_graph / f"{case}.json")
    if (proof["condition"], proof["seed"], proof["test_count"]) != (condition, seed, 10000):
        raise ValueError(f"Wrong cap reference for {case}")
    geometry_path = frozen_graph / "geometry.npz"
    geometry_sha = file_sha256(geometry_path)
    truth_path = study / "inputs" / "test_targets.npy"
    if (geometry_sha != proof["graph_sha256"]
            or file_sha256(truth_path) != BANK_HASHES["test_targets"]
            or tuple(row["budget"] for row in proof["records"]) != BUDGETS):
        raise ValueError(f"Geometry or test inputs changed for {case}")
    truth = np.load(truth_path, allow_pickle=False)
    with np.load(geometry_path, allow_pickle=False) as geometry:
        neighbors, heldout = geometry["neighbors"], geometry["heldout"]
    snapshots = study / "runs" / case / "fits"
    anchor_file = snapshots / "02000" / "fit_inputs.npz"
    if file_sha256(anchor_file) != proof["records"][-1]["input_sha256"]:
        raise ValueError(f"Top budget changed for {case}")
    with np.load(anchor_file, allow_pickle=False) as anchor_data:
        anchors = anchor_data["acquired_indices"]
    paths = six_edge_counts(path_matrix(neighbors, len(neighbors)),
                            path_matrix(heldout, len(neighbors)), anchors)
    rows = []
    for reference in proof["records"]:
        budget = reference["budget"]
        snapshot = snapshots / f"{budget:05}" / "fit_inputs.npz"
        if file_sha256(snapshot) != reference["input_sha256"]:
            raise ValueError(f"Official episode changed for {case} at {budget}")
        with np.load(snapshot, allow_pickle=False) as frozen:
            keep = frozen["base_keep"]
            labels = frozen["observed_labels"][keep]
            aum = frozen["aum_scores"][keep]
        weights = class_weights(labels, aum)
        np.testing.assert_allclose(np.bincount(labels, weights=weights), np.bincount(labels), atol=1e-10)
        accepted = paths[:, :budget][:, keep]
        if int((predict(accepted, labels) == truth).sum()) != reference["raw"]:
            raise ValueError(f"Raw graph votes did not reproduce {case} at {budget}")
        rows.append(dict(reference,
                         sqrt=int((predict(np.sqrt(accepted), labels) == truth).sum()),
                         rank=int((predict(accepted * weights, labels) == truth).sum())))
    data = dict(condition=condition, seed=seed, test_count=len(truth), rows=rows,
                source_sha256=file_sha256(Path(__file__)), reference_sha256=file_sha256(frozen_graph / f"{case}.json"))
    with (output / f"{case}.json").open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
    print(f"Verified raw six-hop and evaluated weight variants for {case}", flush=True)


def collect(source, frozen_graph, output):
    source_sha = file_sha256(Path(__file__))
    records = []
    by_setting = {}
    for condition, seed in CASES:
        case = f"{condition}_s{seed}"
        data = read(source / f"{case}.json")
        reference_path = frozen_graph / f"{case}.json"
        if ((data["condition"], data["seed"], data["test_count"]) != (condition, seed, 10000)
                or data["source_sha256"] != source_sha
                or data["reference_sha256"] != file_sha256(reference_path)
                or tuple(row["budget"] for row in data["rows"]) != BUDGETS):
            raise ValueError(f"Wrong replay results for {case}")
        for row, reference in zip(data["rows"], read(reference_path)["records"]):
            if any(row[field] != reference[field] for field in reference):
                raise ValueError(f"Frozen comparator changed in {case}")
            if any(not isinstance(row[field], int) or not 0 <= row[field] <= 10000
                   for field in ("sqrt", "rank")):
                raise ValueError(f"Invalid test count in {case}")
            record = dict(condition=condition, seed=seed, budget=row["budget"],
                          sqrt_pct=row["sqrt"] / 100, rank_pct=row["rank"] / 100, raw_pct=row["raw"] / 100,
                          delta_sqrt_pp=(row["sqrt"] - row["raw"]) / 100,
                          delta_rank_pp=(row["rank"] - row["raw"]) / 100)
            records.append(record)
            by_setting.setdefault((condition, row["budget"]), []).append(record)
    averages = [dict(condition=condition, budget=budget, seeds=len(values),
                     **{key: mean(v[key] for v in values) for key in
                        ("sqrt_pct", "rank_pct", "raw_pct", "delta_sqrt_pp", "delta_rank_pp")})
                for (condition, budget), values in by_setting.items()]
    output.mkdir(parents=True)
    for name, rows in (("per_seed.csv", records), ("by_setting.csv", averages)):
        with (output / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(f"Verified {len(records)} pairs over {len(averages)} settings")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    run = sub.add_parser("score")
    run.add_argument("study", type=Path)
    run.add_argument("frozen_graph", type=Path)
    run.add_argument("task", type=int, choices=range(len(CASES)))
    run.add_argument("output", type=Path)
    summary = sub.add_parser("collect")
    summary.add_argument("source", type=Path)
    summary.add_argument("frozen_graph", type=Path)
    summary.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.phase == "score":
        score(args.study, args.frozen_graph, args.task, args.output)
    else:
        collect(args.source, args.frozen_graph, args.output)


if __name__ == "__main__":
    main()
