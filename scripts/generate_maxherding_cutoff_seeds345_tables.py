#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


EXP_PATTERN = re.compile(
    r"maxherding_cutoff_(?P<dataset>cifar10|cifar100)_seed(?P<seed>[345])_nn1_"
    r"(?P<sigma>global_sigma|local_sigma_k(?P<k>\d+))_"
)

DATASET_META = {
    "cifar10": {"budget": 10, "max_iter": 9},
    "cifar100": {"budget": 100, "max_iter": 9},
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


def load_accuracy_curve(exp_dir: Path):
    x_path = exp_dir / "plot_episode_xvalues.npy"
    y_path = exp_dir / "plot_episode_yvalues.npy"
    if not x_path.exists() or not y_path.exists():
        return None
    y_vals = np.load(y_path).tolist()
    return [float(v) for v in y_vals]


def load_sigma_stats(exp_dir: Path):
    stats_path = exp_dir / "local_sigma_stats.json"
    if not stats_path.exists():
        return {}
    with stats_path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def collect_experiments(root: Path):
    experiments = []
    for exp_dir in sorted(root.rglob("*")):
        if not exp_dir.is_dir():
            continue
        match = EXP_PATTERN.search(exp_dir.name)
        if not match:
            continue
        y_vals = load_accuracy_curve(exp_dir)
        if y_vals is None:
            continue
        experiments.append(
            {
                "dataset": match.group("dataset"),
                "seed": int(match.group("seed")),
                "sigma_mode": "global_sigma" if match.group("sigma") == "global_sigma" else "local_sigma",
                "local_sigma_k": int(match.group("k")) if match.group("k") else "",
                "exp_dir": str(exp_dir),
                "y_vals": y_vals,
                "sigma_stats": load_sigma_stats(exp_dir),
            }
        )
    return experiments


def labeled_sizes(dataset: str):
    meta = DATASET_META[dataset]
    return [meta["budget"] * (episode + 1) for episode in range(meta["max_iter"] + 1)]


def write_dataset_csv(experiments, dataset: str, out_path: Path):
    sizes = labeled_sizes(dataset)
    fieldnames = ["dataset", "seed", "sigma_mode", "local_sigma_k", "exp_dir"]
    fieldnames.extend([f"labels_{size}" for size in sizes])
    fieldnames.extend(SIGMA_FIELDS)

    rows = []
    dataset_rows = [e for e in experiments if e["dataset"] == dataset]
    dataset_rows.sort(key=lambda item: (item["seed"], item["sigma_mode"] != "global_sigma", item["local_sigma_k"] if item["local_sigma_k"] != "" else -1))
    for exp in dataset_rows:
        row = {
            "dataset": exp["dataset"],
            "seed": exp["seed"],
            "sigma_mode": exp["sigma_mode"],
            "local_sigma_k": exp["local_sigma_k"],
            "exp_dir": exp["exp_dir"],
        }
        for size, value in zip(sizes, exp["y_vals"]):
            row[f"labels_{size}"] = value
        for field in SIGMA_FIELDS:
            row[field] = exp["sigma_stats"].get(field, "")
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate cutoff MaxHerding seeds 3/4/5 summary tables.")
    parser.add_argument("--root", default="output", help="Root output directory to scan")
    parser.add_argument(
        "--outdir",
        default="results_summaries/maxherding_cutoff_seeds345_tables",
        help="Directory for CSV summaries",
    )
    args = parser.parse_args()

    experiments = collect_experiments(Path(args.root))
    outdir = Path(args.outdir)

    targets = [
        ("cifar10", outdir / "cifar10_nn1_cutoff_seeds345.csv"),
        ("cifar100", outdir / "cifar100_nn1_cutoff_seeds345.csv"),
    ]

    for dataset, out_path in targets:
        count = write_dataset_csv(experiments, dataset, out_path)
        print(f"Wrote {count} rows to {out_path}")


if __name__ == "__main__":
    main()
