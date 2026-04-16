#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


PATTERN = re.compile(
    r"probcover_(?P<dataset>cifar10|cifar100)_seed2_(?P<mode>linear|images)_"
    r"(?P<sigma>global_sigma|local_sigma_k(?P<k>\d+))_"
)

DATASET_META = {
    "cifar10": {"budget": 10, "max_iter": 9},
    "cifar100": {"budget": 200, "max_iter": 14},
}

SIGMA_FIELDS = [
    "mean",
    "std",
    "cv",
    "min",
    "q05",
    "q25",
    "median",
    "q75",
    "q95",
    "max",
    "iqr",
    "range",
]


def load_curve(exp_dir: Path):
    x_path = exp_dir / "plot_episode_xvalues.npy"
    y_path = exp_dir / "plot_episode_yvalues.npy"
    if not x_path.exists() or not y_path.exists():
        return None, None
    return np.load(x_path).tolist(), np.load(y_path).tolist()


def load_sigma_stats(exp_dir: Path):
    stats_path = exp_dir / "local_sigma_stats.json"
    if not stats_path.exists():
        return {}
    with stats_path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def collect(root: Path):
    experiments = []
    for exp_dir in sorted(root.rglob("*")):
        if not exp_dir.is_dir():
            continue
        match = PATTERN.search(exp_dir.name)
        if not match:
            continue
        _, y_vals = load_curve(exp_dir)
        if y_vals is None:
            continue
        experiments.append(
            {
                "dataset": match.group("dataset"),
                "mode": match.group("mode"),
                "sigma_name": match.group("sigma"),
                "local_sigma_k": int(match.group("k")) if match.group("k") else "",
                "exp_dir": str(exp_dir),
                "y_vals": y_vals,
                "sigma_stats": load_sigma_stats(exp_dir),
            }
        )
    return experiments


def write_table(experiments, dataset, mode, output_path: Path):
    meta = DATASET_META[dataset]
    labeled_sizes = [meta["budget"] * (episode + 1) for episode in range(meta["max_iter"] + 1)]
    fieldnames = ["dataset", "eval_mode", "sigma_mode", "local_sigma_k", "exp_dir"]
    fieldnames.extend([f"labels_{size}" for size in labeled_sizes])
    fieldnames.extend(SIGMA_FIELDS)

    rows = []
    filtered = [e for e in experiments if e["dataset"] == dataset and e["mode"] == mode]
    filtered.sort(key=lambda item: (item["sigma_name"] != "global_sigma", item["local_sigma_k"] if item["local_sigma_k"] != "" else -1))
    for exp in filtered:
        row = {
            "dataset": dataset,
            "eval_mode": mode,
            "sigma_mode": "global_sigma" if exp["sigma_name"] == "global_sigma" else "local_sigma",
            "local_sigma_k": exp["local_sigma_k"],
            "exp_dir": exp["exp_dir"],
        }
        for size, value in zip(labeled_sizes, exp["y_vals"]):
            row[f"labels_{size}"] = value
        for field in SIGMA_FIELDS:
            row[field] = exp["sigma_stats"].get(field, "")
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate seed-2 ProbCover summary tables.")
    parser.add_argument("--root", default="output")
    parser.add_argument("--outdir", default="results_summaries/probcover_seed2_tables")
    args = parser.parse_args()

    experiments = collect(Path(args.root))
    outdir = Path(args.outdir)
    targets = [
        ("cifar10", "linear", outdir / "cifar10_linear_seed2_probcover.csv"),
        ("cifar10", "images", outdir / "cifar10_images_seed2_probcover.csv"),
        ("cifar100", "linear", outdir / "cifar100_linear_seed2_probcover.csv"),
        ("cifar100", "images", outdir / "cifar100_images_seed2_probcover.csv"),
    ]
    for dataset, mode, output_path in targets:
        count = write_table(experiments, dataset, mode, output_path)
        print(f"Wrote {count} rows to {output_path}")


if __name__ == "__main__":
    main()
