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


def load_curve(exp_dir: Path):
    x_path = exp_dir / "plot_episode_yvalues.npy"
    if not x_path.exists():
        return None
    return [float(v) for v in np.load(x_path).tolist()]


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
        curve = load_curve(exp_dir)
        if curve is None:
            continue
        experiments.append(
            {
                "dataset": match.group("dataset"),
                "seed": int(match.group("seed")),
                "sigma_mode": "global_sigma" if match.group("sigma") == "global_sigma" else "local_sigma",
                "local_sigma_k": int(match.group("k")) if match.group("k") else "",
                "exp_dir": str(exp_dir),
                "curve": curve,
                "sigma_stats": load_sigma_stats(exp_dir),
            }
        )
    return experiments


def labeled_sizes(dataset: str):
    meta = DATASET_META[dataset]
    return [meta["budget"] * (episode + 1) for episode in range(meta["max_iter"] + 1)]


def write_csv(rows, dataset, seed, out_path: Path):
    sizes = labeled_sizes(dataset)
    fieldnames = ["dataset", "seed", "sigma_mode", "local_sigma_k", "exp_dir"]
    fieldnames.extend([f"labels_{size}" for size in sizes])
    fieldnames.extend(SIGMA_FIELDS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {
                "dataset": dataset,
                "seed": seed,
                "sigma_mode": row["sigma_mode"],
                "local_sigma_k": row["local_sigma_k"],
                "exp_dir": row["exp_dir"],
            }
            for size, value in zip(sizes, row["curve"]):
                out[f"labels_{size}"] = value
            for field in SIGMA_FIELDS:
                out[field] = row["sigma_stats"].get(field, "")
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser(description="Generate per-seed cutoff MaxHerding CSV tables.")
    parser.add_argument("--root", default="output")
    parser.add_argument("--outdir", default="results_summaries/maxherding_cutoff_seeds345_tables/per_seed")
    args = parser.parse_args()

    experiments = collect_experiments(Path(args.root))
    outdir = Path(args.outdir)

    for dataset in ["cifar10", "cifar100"]:
        for seed in [3, 4, 5]:
            rows = [e for e in experiments if e["dataset"] == dataset and e["seed"] == seed]
            rows.sort(key=lambda item: (item["sigma_mode"] != "global_sigma", item["local_sigma_k"] if item["local_sigma_k"] != "" else -1))
            out_path = outdir / f"{dataset}_seed{seed}_nn1_cutoff.csv"
            write_csv(rows, dataset, seed, out_path)
            print(out_path)


if __name__ == "__main__":
    main()
