#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/cs/labs/daphna/ido.dotan/projects/daphna_lab")
CSV_DIR = ROOT / "results_summaries" / "probcover_seed2_tables"
VIS_DIR = ROOT / "visual_outputs" / "probcover" / "seed2_final_vs_k"

CSV_FILES = [
    "cifar10_linear_seed2_probcover.csv",
    "cifar10_images_seed2_probcover.csv",
    "cifar100_linear_seed2_probcover.csv",
    "cifar100_images_seed2_probcover.csv",
]


def friendly_mode(mode: str):
    return {
        "linear": "Feature Linear",
        "images": "Image Training",
    }[mode]


def load_rows(csv_path: Path):
    with csv_path.open(newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


def plot_csv(csv_path: Path):
    rows = load_rows(csv_path)
    if not rows:
        return None

    label_cols = [col for col in rows[0] if col.startswith("labels_")]
    final_col = label_cols[-1]
    global_row = next(row for row in rows if row["sigma_mode"] == "global_sigma")
    global_final = float(global_row[final_col])

    local_rows = []
    for row in rows:
        if row["sigma_mode"] != "local_sigma":
            continue
        value = row[final_col]
        if value == "":
            continue
        local_rows.append((int(row["local_sigma_k"]), float(value)))
    local_rows.sort()

    dataset = rows[0]["dataset"]
    mode = rows[0]["eval_mode"]
    plt.figure(figsize=(9.5, 5.8))
    if local_rows:
        ks = [k for k, _ in local_rows]
        finals = [acc for _, acc in local_rows]
        xpos = list(range(len(ks)))
        plt.plot(xpos, finals, marker="o", linewidth=2.2, markersize=5.5, color="#1f77b4", label="Local sigma")
        ymin = min(min(finals), global_final)
        ymax = max(max(finals), global_final)
    else:
        ks = []
        xpos = []
        ymin = ymax = global_final

    plt.axhline(global_final, color="#d62728", linestyle="--", linewidth=2, label=f"Global sigma ({global_final:.2f})")
    margin = max(0.15, (ymax - ymin) * 0.18 if ymax > ymin else 0.5)
    plt.ylim(ymin - margin, ymax + margin)
    if ks:
        plt.xticks(xpos, [str(k) for k in ks], rotation=45, ha="right")
    plt.xlabel("K")
    plt.ylabel("Final test accuracy (%)")
    plt.title(f"{dataset.upper()} {friendly_mode(mode)}: Final Accuracy vs K")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    VIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIS_DIR / f"{dataset}_{mode}_seed2_probcover_final_accuracy_vs_k.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def main():
    generated = []
    for name in CSV_FILES:
        csv_path = CSV_DIR / name
        if not csv_path.exists():
            continue
        out_path = plot_csv(csv_path)
        if out_path is not None:
            generated.append(out_path)
    for out_path in generated:
        print(out_path)


if __name__ == "__main__":
    main()
