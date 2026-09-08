#!/usr/bin/env python3
"""Plot fixed-alpha density-normalization gains over alpha=0.

For each dataset and sigma setting, this scans the saved MaxHerding fixed-alpha
runs and computes:

    gain(alpha, budget) = accuracy(alpha, budget) - accuracy(alpha=0, budget)

The plot averages over every seed that has both the requested alpha and a
matching alpha=0 baseline. In the current experiments, alpha 0.25 and 0.5 have
seeds 3/4/5, while alpha 0.75 has seed 3 only.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_ROOT = Path("output")
RESULTS_DIR = Path("results_summaries/maxherding_density_norm/alpha_gain")
PLOTS_DIR = Path("visual_outputs/maxherding/density_norm/alpha_gain")

TARGET_ALPHAS = (0.25, 0.5, 0.75)
BEST_LOCAL_K = {
    "CIFAR10": "k=3500",
    "CIFAR100": "k=1440",
}

RUN_RE = re.compile(
    r"maxherding_dnorm(?:_alpha_wide)?_"
    r"(?P<dataset>cifar10|cifar100)_seed(?P<seed>\d+)_nn1_"
    r"(?P<sigma>global|local_sigma_k(?P<k>\d+))_"
    r"alpha(?P<alpha>[^_]+)_weighted"
)


def dataset_name(raw: str) -> str:
    return "CIFAR10" if raw == "cifar10" else "CIFAR100"


def setting_name(sigma: str, k: str | None) -> str:
    return "global" if sigma == "global" else f"k={int(k)}"


def labels_for(dataset: str, episode: int) -> int:
    step = 10 if dataset == "CIFAR10" else 100
    return step * (episode + 1)


def load_curve(run_dir: Path) -> list[float]:
    return [float(x) for x in np.load(run_dir / "plot_episode_yvalues.npy")]


def collect_runs() -> pd.DataFrame:
    rows: list[dict] = []
    for run_dir in OUTPUT_ROOT.glob("CIFAR*/resnet18/**/maxherding_dnorm*"):
        if not run_dir.is_dir():
            continue
        # Exclude scheduled/decay experiments. We only want fixed-alpha kernels.
        if "decay" in run_dir.name or "sched" in run_dir.name:
            continue
        match = RUN_RE.search(run_dir.name)
        if not match:
            continue

        dataset = dataset_name(match.group("dataset"))
        setting = setting_name(match.group("sigma"), match.group("k"))
        if setting not in {"global", BEST_LOCAL_K[dataset]}:
            continue

        alpha = float(match.group("alpha"))
        if alpha not in {0.0, *TARGET_ALPHAS}:
            continue

        curve_path = run_dir / "plot_episode_yvalues.npy"
        if not curve_path.exists():
            continue

        for episode, accuracy in enumerate(load_curve(run_dir)):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": int(match.group("seed")),
                    "setting": setting,
                    "alpha": alpha,
                    "episode": episode,
                    "labels": labels_for(dataset, episode),
                    "accuracy": accuracy,
                    "run": str(run_dir),
                    "mtime": run_dir.stat().st_mtime,
                }
            )

    if not rows:
        raise RuntimeError("No fixed-alpha weighted density-normalization runs found.")

    df = pd.DataFrame(rows)
    # Some seed/configs were rerun. Keep the newest physical run for each logical
    # result to avoid double-counting duplicates.
    return (
        df.sort_values("mtime")
        .drop_duplicates(
            ["dataset", "seed", "setting", "alpha", "episode"],
            keep="last",
        )
        .sort_values(["dataset", "seed", "setting", "alpha", "episode"])
    )


def compute_gains(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = runs[runs["alpha"] == 0.0][
        ["dataset", "seed", "setting", "episode", "accuracy"]
    ].rename(columns={"accuracy": "baseline_accuracy"})
    gains = runs[runs["alpha"].isin(TARGET_ALPHAS)].merge(
        baseline,
        on=["dataset", "seed", "setting", "episode"],
        how="inner",
    )
    gains["gain_vs_alpha0"] = gains["accuracy"] - gains["baseline_accuracy"]

    mean = (
        gains.groupby(["dataset", "setting", "alpha", "episode", "labels"], as_index=False)
        .agg(
            mean_gain_vs_alpha0=("gain_vs_alpha0", "mean"),
            std_gain_vs_alpha0=("gain_vs_alpha0", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_baseline_accuracy=("baseline_accuracy", "mean"),
            seeds=("seed", "nunique"),
            seed_list=("seed", lambda s: ";".join(str(x) for x in sorted(set(s)))),
        )
        .sort_values(["dataset", "setting", "alpha", "episode"])
    )
    return gains, mean


def plot_one_axis(ax: plt.Axes, data: pd.DataFrame, title: str) -> None:
    for alpha, group in data.groupby("alpha"):
        group = group.sort_values("labels")
        ax.plot(
            group["labels"],
            group["mean_gain_vs_alpha0"],
            marker="o",
            linewidth=2,
            label=f"alpha={alpha:g}",
        )
        seed_count = int(group["seeds"].max())
        if seed_count > 1:
            y = group["mean_gain_vs_alpha0"].to_numpy()
            std = group["std_gain_vs_alpha0"].fillna(0).to_numpy()
            ax.fill_between(group["labels"], y - std, y + std, alpha=0.12)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Labeled points")
    ax.set_ylabel("Accuracy gain over alpha=0")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def write_plots(mean: pd.DataFrame) -> None:
    for dataset in sorted(mean["dataset"].unique()):
        dataset_part = mean[mean["dataset"] == dataset]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, setting in zip(axes, ["global", BEST_LOCAL_K[dataset]]):
            part = dataset_part[dataset_part["setting"] == setting]
            plot_one_axis(ax, part, f"{dataset} {setting}")
        fig.suptitle(f"{dataset}: fixed-alpha density-normalization gain")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{dataset.lower()}_global_and_bestk_alpha_gain.png", dpi=170)
        plt.close(fig)

        for setting in ["global", BEST_LOCAL_K[dataset]]:
            part = dataset_part[dataset_part["setting"] == setting]
            fig, ax = plt.subplots(figsize=(8.5, 5))
            plot_one_axis(ax, part, f"{dataset} {setting}: alpha gain over alpha=0")
            fig.tight_layout()
            fig.savefig(
                PLOTS_DIR / f"{dataset.lower()}_{setting.replace('=', '')}_alpha_gain.png",
                dpi=170,
            )
            plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    runs = collect_runs()
    gains, mean = compute_gains(runs)

    runs.drop(columns=["mtime"]).to_csv(RESULTS_DIR / "fixed_alpha_runs_by_episode.csv", index=False)
    gains.drop(columns=["mtime"]).to_csv(RESULTS_DIR / "alpha_gain_by_seed.csv", index=False)
    mean.to_csv(RESULTS_DIR / "alpha_gain_mean.csv", index=False)

    write_plots(mean)

    print(f"Wrote tables to {RESULTS_DIR}")
    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
