#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/cs/labs/daphna/ido.dotan/projects/daphna_lab")
CSV_DIR = ROOT / "results_summaries" / "maxherding_cutoff_seeds345_tables"
VIS_DIR = ROOT / "visual_outputs" / "maxherding" / "cutoff_seeds345"

CSV_FILES = [
    ("cifar10", CSV_DIR / "cifar10_nn1_cutoff_seeds345.csv"),
    ("cifar100", CSV_DIR / "cifar100_nn1_cutoff_seeds345.csv"),
]


def load_rows(csv_path: Path):
    with csv_path.open(newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


def parse_dataset(rows):
    label_cols = [col for col in rows[0] if col.startswith("labels_")]
    labeled_sizes = [int(col.split("_", 1)[1]) for col in label_cols]
    grouped = defaultdict(dict)
    for row in rows:
        seed = int(row["seed"])
        sigma_mode = row["sigma_mode"]
        key = "global" if sigma_mode == "global_sigma" else int(row["local_sigma_k"])
        values = []
        for col in label_cols:
            raw = row[col]
            values.append(float(raw) if raw != "" else np.nan)
        grouped[key][seed] = np.array(values, dtype=float)
    return labeled_sizes, grouped


def mean_std(curves_by_seed):
    seeds = sorted(curves_by_seed)
    stacked = np.stack([curves_by_seed[s] for s in seeds], axis=0)
    return np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0), seeds


def plot_final_vs_k(dataset, grouped):
    local_keys = sorted(k for k in grouped.keys() if k != "global")
    global_mean, global_std, _ = mean_std(grouped["global"])

    xs = local_keys
    local_finals = []
    local_stds = []
    for k in local_keys:
        mean_curve, std_curve, _ = mean_std(grouped[k])
        local_finals.append(mean_curve[-1])
        local_stds.append(std_curve[-1])

    plt.figure(figsize=(9.5, 5.8))
    plt.errorbar(xs, local_finals, yerr=local_stds, marker="o", linewidth=2.2, capsize=4, label="Local sigma")
    plt.axhline(global_mean[-1], color="#d62728", linestyle="--", linewidth=2, label=f"Global mean ({global_mean[-1]:.2f})")
    plt.fill_between([min(xs), max(xs)], global_mean[-1] - global_std[-1], global_mean[-1] + global_std[-1], color="#d62728", alpha=0.12)

    ymin = min(min(local_finals), global_mean[-1]) - 0.8
    ymax = max(max(local_finals), global_mean[-1]) + 0.8
    plt.ylim(ymin, ymax)
    plt.xticks(xs, [str(x) for x in xs], rotation=45, ha="right")
    plt.xlabel("K")
    plt.ylabel("Final test accuracy (%)")
    plt.title(f"{dataset.upper()} MaxHerding cutoff: final accuracy vs K (seeds 3-5)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_path = VIS_DIR / f"{dataset}_nn1_cutoff_seeds345_final_accuracy_vs_k.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def plot_selected_curves(dataset, labeled_sizes, grouped):
    global_mean, global_std, _ = mean_std(grouped["global"])
    local_keys = sorted(k for k in grouped.keys() if k != "global")

    plt.figure(figsize=(9.8, 6.0))
    plt.plot(labeled_sizes, global_mean, linewidth=2.4, color="#d62728", label="Global")
    plt.fill_between(labeled_sizes, global_mean - global_std, global_mean + global_std, color="#d62728", alpha=0.10)

    cmap = plt.get_cmap("viridis")
    for idx, k in enumerate(local_keys):
        mean_curve, std_curve, _ = mean_std(grouped[k])
        color = cmap(idx / max(1, len(local_keys) - 1))
        plt.plot(labeled_sizes, mean_curve, linewidth=1.8, color=color, label=f"K={k}")

    all_means = [global_mean] + [mean_std(grouped[k])[0] for k in local_keys]
    ymin = min(np.nanmin(curve) for curve in all_means) - 1.0
    ymax = max(np.nanmax(curve) for curve in all_means) + 1.0
    plt.ylim(ymin, ymax)
    plt.xlabel("Number of labeled points")
    plt.ylabel("Test accuracy (%)")
    plt.title(f"{dataset.upper()} MaxHerding cutoff: global vs selected K curves (seeds 3-5 mean)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    out_path = VIS_DIR / f"{dataset}_nn1_cutoff_seeds345_all_k_curves.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def plot_best_vs_global(dataset, labeled_sizes, grouped):
    global_mean, global_std, _ = mean_std(grouped["global"])
    best_k = None
    best_final = -np.inf
    best_curve = None
    best_std = None

    for k in sorted(k for k in grouped.keys() if k != "global"):
        mean_curve, std_curve, _ = mean_std(grouped[k])
        if mean_curve[-1] > best_final:
            best_final = mean_curve[-1]
            best_k = k
            best_curve = mean_curve
            best_std = std_curve

    plt.figure(figsize=(8.8, 5.8))
    plt.plot(labeled_sizes, global_mean, linewidth=2.4, color="#d62728", label="Global")
    plt.fill_between(labeled_sizes, global_mean - global_std, global_mean + global_std, color="#d62728", alpha=0.12)
    plt.plot(labeled_sizes, best_curve, linewidth=2.4, color="#1f77b4", label=f"Best local K={best_k}")
    plt.fill_between(labeled_sizes, best_curve - best_std, best_curve + best_std, color="#1f77b4", alpha=0.12)

    ymin = min(np.nanmin(global_mean - global_std), np.nanmin(best_curve - best_std)) - 0.8
    ymax = max(np.nanmax(global_mean + global_std), np.nanmax(best_curve + best_std)) + 0.8
    plt.ylim(ymin, ymax)
    plt.xlabel("Number of labeled points")
    plt.ylabel("Test accuracy (%)")
    plt.title(f"{dataset.upper()} MaxHerding cutoff: best local K vs global (seeds 3-5 mean)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_path = VIS_DIR / f"{dataset}_nn1_cutoff_seeds345_best_k_vs_global.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def main():
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    for dataset, csv_path in CSV_FILES:
        if not csv_path.exists():
            continue
        rows = load_rows(csv_path)
        if not rows:
            continue
        labeled_sizes, grouped = parse_dataset(rows)
        generated.append(plot_final_vs_k(dataset, grouped))
        generated.append(plot_selected_curves(dataset, labeled_sizes, grouped))
        generated.append(plot_best_vs_global(dataset, labeled_sizes, grouped))
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
