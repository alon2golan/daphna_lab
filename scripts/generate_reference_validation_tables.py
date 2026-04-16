#!/usr/bin/env python3
import csv
import re
from pathlib import Path

import numpy as np


ROOT = Path("/cs/labs/daphna/ido.dotan/projects/daphna_lab")
OUTPUT_ROOT = ROOT / "output"
OUTDIR = ROOT / "results_summaries" / "reference_validation_tables"

PATTERN = re.compile(
    r"refval_(?P<method>probcover|maxherding)_(?P<dataset>cifar10|cifar100)_seed(?P<seed>\d+)_"
    r"(?P<mode>paper_images|paper_nn1|nn1|logistic)_global_"
)

CONFIGS = [
    ("probcover", "cifar10", "cifar10_probcover_global_reference_combined.csv"),
    ("probcover", "cifar100", "cifar100_probcover_global_reference_combined.csv"),
    ("maxherding", "cifar10", "cifar10_maxherding_global_reference_combined.csv"),
    ("maxherding", "cifar100", "cifar100_maxherding_global_reference_combined.csv"),
]

PAPER_RESULTS = {
    ("probcover", "cifar10"): {
        10: 12.62,
        20: 13.30,
        30: 13.91,
        40: 14.32,
        50: 15.09,
        200: 24.58,
        400: 46.90,
        600: 58.34,
        800: 65.90,
        1400: 78.36,
        2600: 86.31,
        3800: 89.28,
        5000: 91.50,
        6200: 92.54,
        7400: 93.47,
    },
    ("probcover", "cifar100"): {
        200: 10.79,
        400: 13.53,
        600: 15.86,
        1000: 19.49,
        3000: 29.31,
        5000: 34.64,
        7000: 38.36,
        11000: 44.69,
        15000: 49.58,
        19000: 52.90,
        23000: 56.29,
    },
}

MODE_LABELS = {
    ("probcover", "paper_images"): "from_images",
    ("probcover", "nn1"): "nn1",
    ("probcover", "logistic"): "logistic",
    ("maxherding", "paper_nn1"): "nn1",
    ("maxherding", "logistic"): "logistic",
}


def load_curve(exp_dir: Path, dataset: str):
    x_path = exp_dir / "plot_episode_xvalues.npy"
    y_path = exp_dir / "plot_episode_yvalues.npy"
    if not x_path.exists() or not y_path.exists():
        return None
    x_vals = np.load(x_path).tolist()
    y_vals = np.load(y_path).tolist()
    x_vals = [int(round(x)) for x in x_vals]
    y_vals = [float(y) for y in y_vals]

    expected = expected_sizes(dataset)
    if x_vals == list(range(len(x_vals))):
        mapped_sizes = expected[: len(x_vals)]
        return list(zip(mapped_sizes, y_vals))

    return list(zip(x_vals, y_vals))


def collect_runs():
    runs = {}
    for exp_dir in sorted(OUTPUT_ROOT.rglob("*")):
        if not exp_dir.is_dir():
            continue
        match = PATTERN.search(exp_dir.name)
        if not match:
            continue
        curve = load_curve(exp_dir, match.group("dataset"))
        if curve is None:
            continue
        key = (match.group("method"), match.group("dataset"), match.group("mode"))
        seed = int(match.group("seed"))
        runs.setdefault(key, {})[seed] = {
            "exp_dir": str(exp_dir.relative_to(ROOT)),
            "curve": curve,
        }
    return runs


def expected_sizes(dataset: str):
    if dataset == "cifar10":
        return list(range(10, 101, 10))
    return list(range(100, 1001, 100))


def write_table(method, dataset, method_runs, filename):
    sizes = expected_sizes(dataset)
    paper_map = PAPER_RESULTS.get((method, dataset), {})
    mode_order = [mode for mode in ["paper_images", "paper_nn1", "nn1", "logistic"] if mode in method_runs]
    fieldnames = ["labeled_set_size", "paper_result"]
    per_mode_seed_values = {}
    for mode in mode_order:
        fieldnames.extend([f"{MODE_LABELS[(method, mode)]}_seed_0", f"{MODE_LABELS[(method, mode)]}_seed_1", f"{MODE_LABELS[(method, mode)]}_seed_2"])
        seeds_data = method_runs.get(mode, {})
        per_mode_seed_values[mode] = {}
        for seed in [0, 1, 2]:
            run = seeds_data.get(seed)
            per_mode_seed_values[mode][seed] = {} if run is None else dict(run["curve"])

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / filename
    with out_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for size in sizes:
            row = {
                "labeled_set_size": size,
                "paper_result": paper_map.get(size, ""),
            }
            for mode in mode_order:
                label = MODE_LABELS[(method, mode)]
                for seed in [0, 1, 2]:
                    row[f"{label}_seed_{seed}"] = per_mode_seed_values[mode][seed].get(size, "")
            writer.writerow(row)
    return out_path


def main():
    runs = collect_runs()
    generated = []
    for method, dataset, filename in CONFIGS:
        method_runs = {}
        for run_key, seeds_data in runs.items():
            run_method, run_dataset, mode = run_key
            if run_method == method and run_dataset == dataset:
                method_runs[mode] = seeds_data
        generated.append(write_table(method, dataset, method_runs, filename))
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
