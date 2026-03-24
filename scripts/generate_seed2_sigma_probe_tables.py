#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


K_PATTERN = re.compile(
    r"maxherding_(?P<dataset>cifar10|cifar100)_seed2_(?P<mode>nn1|linear)_"
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


def load_accuracy_curve(exp_dir: Path):
    x_path = exp_dir / "plot_episode_xvalues.npy"
    y_path = exp_dir / "plot_episode_yvalues.npy"
    if not x_path.exists() or not y_path.exists():
        return None, None
    x_vals = np.load(x_path)
    y_vals = np.load(y_path)
    return x_vals.tolist(), y_vals.tolist()


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
        match = K_PATTERN.search(exp_dir.name)
        if not match:
            continue
        dataset = match.group("dataset")
        mode = match.group("mode")
        sigma_name = match.group("sigma")
        local_k = match.group("k")
        x_vals, y_vals = load_accuracy_curve(exp_dir)
        if y_vals is None:
            continue
        sigma_stats = load_sigma_stats(exp_dir)
        experiments.append(
            {
                "dataset": dataset,
                "mode": mode,
                "sigma_name": sigma_name,
                "local_sigma_k": int(local_k) if local_k is not None else "",
                "exp_dir": str(exp_dir),
                "x_vals": x_vals,
                "y_vals": y_vals,
                "sigma_stats": sigma_stats,
            }
        )
    return experiments


def write_table(experiments, dataset, mode, output_path: Path):
    meta = DATASET_META[dataset]
    budget = meta["budget"]
    max_iter = meta["max_iter"]
    labeled_sizes = [budget * (episode + 1) for episode in range(max_iter + 1)]

    fieldnames = [
        "dataset",
        "eval_mode",
        "sigma_mode",
        "local_sigma_k",
        "exp_dir",
    ]
    fieldnames.extend([f"labels_{size}" for size in labeled_sizes])
    fieldnames.extend(SIGMA_FIELDS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for exp in sorted(
        [e for e in experiments if e["dataset"] == dataset and e["mode"] == mode],
        key=lambda item: (item["sigma_name"] != "global_sigma", item["local_sigma_k"] if item["local_sigma_k"] != "" else -1),
    ):
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

    with output_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate seed-2 sigma-probe summary tables.")
    parser.add_argument("--root", default="output", help="Root output directory to scan")
    parser.add_argument(
        "--outdir",
        default="results_summaries/seed2_sigma_probe_tables",
        help="Directory for CSV summaries",
    )
    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    experiments = collect_experiments(root)

    targets = [
        ("cifar10", "nn1", outdir / "cifar10_nn1_seed2_sigma_probe.csv"),
        ("cifar10", "linear", outdir / "cifar10_linear_seed2_sigma_probe.csv"),
        ("cifar100", "nn1", outdir / "cifar100_nn1_seed2_sigma_probe.csv"),
        ("cifar100", "linear", outdir / "cifar100_linear_seed2_sigma_probe.csv"),
    ]

    for dataset, mode, output_path in targets:
        count = write_table(experiments, dataset, mode, output_path)
        print(f"Wrote {count} rows to {output_path}")


if __name__ == "__main__":
    main()
