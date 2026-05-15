#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/cs/labs/daphna/ido.dotan/projects/daphna_lab")
CSV_DIR = ROOT / "results_summaries" / "maxherding_cutoff_seeds345_tables" / "per_seed"
VIS_DIR = ROOT / "visual_outputs" / "maxherding" / "cutoff_seeds345" / "per_seed"


def load_rows(path: Path):
    with path.open(newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


def plot_final_vs_k(rows, dataset, seed):
    label_cols = [c for c in rows[0] if c.startswith("labels_")]
    final_col = label_cols[-1]
    global_row = next(row for row in rows if row["sigma_mode"] == "global_sigma")
    global_final = float(global_row[final_col])
    local = []
    for row in rows:
        if row["sigma_mode"] != "local_sigma":
            continue
        local.append((int(row["local_sigma_k"]), float(row[final_col])))
    local.sort()

    xs = [k for k, _ in local]
    ys = [v for _, v in local]

    plt.figure(figsize=(9.2, 5.6))
    plt.plot(xs, ys, marker="o", linewidth=2.2, markersize=5.5, label="Local sigma")
    plt.axhline(global_final, color="#d62728", linestyle="--", linewidth=2, label=f"Global ({global_final:.2f})")
    ymin = min(min(ys), global_final) - 0.8
    ymax = max(max(ys), global_final) + 0.8
    plt.ylim(ymin, ymax)
    plt.xticks(xs, [str(x) for x in xs], rotation=45, ha="right")
    plt.xlabel("K")
    plt.ylabel("Final test accuracy (%)")
    plt.title(f"{dataset.upper()} cutoff MaxHerding, seed {seed}: final accuracy vs K")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out = VIS_DIR / f"{dataset}_seed{seed}_final_accuracy_vs_k.png"
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_all_k_curves(rows, dataset, seed):
    label_cols = [c for c in rows[0] if c.startswith("labels_")]
    sizes = [int(c.split("_", 1)[1]) for c in label_cols]
    global_row = next(row for row in rows if row["sigma_mode"] == "global_sigma")
    global_vals = [float(global_row[c]) for c in label_cols]

    plt.figure(figsize=(9.8, 5.9))
    plt.plot(sizes, global_vals, linewidth=2.4, color="#d62728", label="Global")

    cmap = plt.get_cmap("viridis")
    local_rows = sorted(
        [row for row in rows if row["sigma_mode"] == "local_sigma"],
        key=lambda r: int(r["local_sigma_k"]),
    )
    for idx, row in enumerate(local_rows):
        vals = [float(row[c]) for c in label_cols]
        k = int(row["local_sigma_k"])
        color = cmap(idx / max(1, len(local_rows) - 1))
        plt.plot(sizes, vals, linewidth=1.8, color=color, label=f"K={k}")

    all_vals = [global_vals] + [[float(r[c]) for c in label_cols] for r in local_rows]
    ymin = min(min(v) for v in all_vals) - 1.0
    ymax = max(max(v) for v in all_vals) + 1.0
    plt.ylim(ymin, ymax)
    plt.xlabel("Number of labeled points")
    plt.ylabel("Test accuracy (%)")
    plt.title(f"{dataset.upper()} cutoff MaxHerding, seed {seed}: global vs local K curves")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    out = VIS_DIR / f"{dataset}_seed{seed}_all_k_curves.png"
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_best_vs_global(rows, dataset, seed):
    label_cols = [c for c in rows[0] if c.startswith("labels_")]
    sizes = [int(c.split("_", 1)[1]) for c in label_cols]
    global_row = next(row for row in rows if row["sigma_mode"] == "global_sigma")
    global_vals = [float(global_row[c]) for c in label_cols]

    best_row = max(
        [row for row in rows if row["sigma_mode"] == "local_sigma"],
        key=lambda r: float(r[label_cols[-1]]),
    )
    best_vals = [float(best_row[c]) for c in label_cols]
    best_k = int(best_row["local_sigma_k"])

    plt.figure(figsize=(8.8, 5.7))
    plt.plot(sizes, global_vals, linewidth=2.4, color="#d62728", label="Global")
    plt.plot(sizes, best_vals, linewidth=2.4, color="#1f77b4", label=f"Best local K={best_k}")
    ymin = min(min(global_vals), min(best_vals)) - 0.8
    ymax = max(max(global_vals), max(best_vals)) + 0.8
    plt.ylim(ymin, ymax)
    plt.xlabel("Number of labeled points")
    plt.ylabel("Test accuracy (%)")
    plt.title(f"{dataset.upper()} cutoff MaxHerding, seed {seed}: best local K vs global")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out = VIS_DIR / f"{dataset}_seed{seed}_best_k_vs_global.png"
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def main():
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    for dataset in ["cifar10", "cifar100"]:
        for seed in [3, 4, 5]:
            path = CSV_DIR / f"{dataset}_seed{seed}_nn1_cutoff.csv"
            if not path.exists():
                continue
            rows = load_rows(path)
            if not rows:
                continue
            generated.append(plot_final_vs_k(rows, dataset, seed))
            generated.append(plot_all_k_curves(rows, dataset, seed))
            generated.append(plot_best_vs_global(rows, dataset, seed))
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
