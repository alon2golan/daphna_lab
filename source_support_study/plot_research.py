"""Render paired audit and noise-channel results from the saved aggregate files."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def paired_bars(ax, series, groups, title, ylabel):
    positions = np.arange(len(groups))
    offsets = (-.17, .17)
    for (label, values, color), offset in zip(series, offsets):
        means = np.mean(values, axis=1)
        ax.bar(positions + offset, means, .30, label=label, color=color, alpha=.8, zorder=2)
        for position, points in zip(positions + offset, values):
            jitter = np.linspace(-.065, .065, len(points))
            ax.scatter(position + jitter, points, s=25, color=color,
                       edgecolors="white", linewidths=.55, zorder=3)
    ax.axhline(0, color="#475569", lw=.8, zorder=1)
    ax.grid(axis="y", color="#e2e8f0", lw=.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(positions, groups)
    ax.tick_params(axis="both", length=0, pad=7)
    ax.set_ylabel(ylabel, labelpad=9)
    ax.set_title(title, loc="left", fontweight="bold", pad=11)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.margins(y=.22)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    with (here.parent / "active_audit_study/signed_metrics.csv").open() as stream:
        audit = {(r["condition"], r["policy"], int(r["seed"])): int(r["test_correct"])
                 for r in csv.DictReader(stream)}
    channels = json.loads((here / "channel_results.json").read_text())
    channel = {(r["channel"], r["seed"]): r["arms"] for run in channels for r in run["rows"]}
    audit_settings = ("sym50", "asym50", "cifar100n")
    channel_settings = ("uniform", "scattered_cycle", "superclass_cycle")

    def audit_delta(policy):
        return [[(audit[c, policy, s] - audit[c, "loo_audit", s]) / 100
                 for s in range(5, 10)] for c in audit_settings]

    def precision_delta(policy):
        return [[100 * (channel[c, s][policy]["precision"] - channel[c, s]["reference"]["precision"])
                 for s in range(5)] for c in channel_settings]

    plt.rcParams.update({"font.size": 10, "text.color": "#17263a",
                         "axes.labelcolor": "#25364a", "xtick.color": "#25364a",
                         "ytick.color": "#25364a", "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), layout="constrained")
    fig.suptitle("Two tests of annotation evidence on CIFAR-100 BYOL", fontsize=16,
                 fontweight="bold", x=.018, ha="left")
    gray, green, blue = "#678199", "#087f78", "#385f9d"
    paired_bars(
        axes[0], [("Unsigned footprint", audit_delta("single_footprint"), gray),
                  ("Signed audit score", audit_delta("signed_peer"), green)],
        ["Symmetric\n50%", "Asymmetric\n50%", "CIFAR-100N"],
        "A   800 noisy labels + 80 clean requests", "Test accuracy vs leave-one-out (pp)",
    )
    paired_bars(
        axes[1], [("Top-two margin", precision_delta("margin"), gray),
                  ("Remove strongest source", precision_delta("source_stability"), blue)],
        ["Uniform\nwrong label", "Fixed cycle\nacross classes", "Fixed cycle\nwithin superclass"],
        "B   Same images corrupted at 50%", "Pseudo-label precision vs confidence (pp)",
    )
    fig.supxlabel("Bars show means. Dots show the five paired seeds in each condition.",
                  fontsize=9, x=.02, ha="left")
    fig.savefig(args.output, dpi=180, facecolor="white")


if __name__ == "__main__":
    main()
