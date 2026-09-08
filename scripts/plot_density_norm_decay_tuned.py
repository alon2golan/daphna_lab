#!/usr/bin/env python3
"""Summarize and plot tuned density-normalization schedule experiments.

The tuned schedule runs use names like:

``maxherding_dnorm_decay_cifar10_seed3_nn1_local_sigma_k3500_a075_decay5_weighted``

For interpretation, this script also loads matching ``alpha=0`` weighted
baselines from the earlier density-normalization sweeps and computes deltas
against the same dataset/seed/sigma setting.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_ROOT = Path("output")
RESULTS_DIR = Path("results_summaries/maxherding_density_norm/decay_tuned")
PLOTS_DIR = Path("visual_outputs/maxherding/density_norm/decay_tuned")

DECAY_RE = re.compile(
    r"maxherding_dnorm_decay_"
    r"(?P<dataset>cifar10|cifar100)_seed(?P<seed>\d+)_nn1_"
    r"(?P<sigma>global|local_sigma_k(?P<k>\d+))_"
    r"(?P<tag>a(?P<alpha_tag>\d+)_"
    r"(?P<schedule>early|threefloor|three|decay)(?P<steps>\d+))_weighted",
)

BASELINE_RE = re.compile(
    r"maxherding_dnorm(?:_alpha_wide)?_"
    r"(?P<dataset>cifar10|cifar100)_seed(?P<seed>\d+)_nn1_"
    r"(?P<sigma>global|local_sigma_k(?P<k>\d+))_alpha0_weighted",
)


def dataset_name(raw: str) -> str:
    return "CIFAR10" if raw == "cifar10" else "CIFAR100"


def labels_for(dataset: str, episode: int) -> int:
    budget = 10 if dataset == "CIFAR10" else 100
    return budget * (episode + 1)


def parse_alpha(alpha_tag: str) -> float:
    # a075 -> 0.75, a05 -> 0.5
    return int(alpha_tag) / (10 ** (len(alpha_tag) - 1))


def schedule_label(schedule: str, steps: str) -> str:
    if schedule == "decay":
        return f"linear_decay_steps_{steps}"
    if schedule == "early":
        return f"early_only_{steps}"
    if schedule == "three":
        return f"three_phase_{steps}"
    if schedule == "threefloor":
        return f"three_phase_floor_{steps}"
    raise ValueError(f"Unknown schedule tag: {schedule}")


def setting_label(match: re.Match[str]) -> str:
    if match.group("sigma") == "global":
        return "global"
    return f"k={int(match.group('k'))}"


def load_curve(run_dir: Path) -> list[float]:
    y_path = run_dir / "plot_episode_yvalues.npy"
    if not y_path.exists():
        raise FileNotFoundError(f"Missing accuracy curve: {y_path}")
    return [float(x) for x in np.load(y_path)]


def newest_unique(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # If a run was accidentally repeated, keep the newest directory for the
    # logical configuration.
    return (
        df.sort_values("mtime")
        .drop_duplicates(["dataset", "seed", "setting", "schedule", "alpha", "episode"], keep="last")
        .sort_values(["dataset", "seed", "setting", "schedule", "alpha", "episode"])
    )


def collect_decay_runs() -> pd.DataFrame:
    rows: list[dict] = []
    for run_dir in OUTPUT_ROOT.glob("CIFAR*/resnet18/**/maxherding_dnorm_decay_*"):
        if not run_dir.is_dir():
            continue
        match = DECAY_RE.search(run_dir.name)
        if not match:
            continue
        dataset = dataset_name(match.group("dataset"))
        seed = int(match.group("seed"))
        setting = setting_label(match)
        alpha = parse_alpha(match.group("alpha_tag"))
        schedule = schedule_label(match.group("schedule"), match.group("steps"))
        curve = load_curve(run_dir)
        for episode, accuracy in enumerate(curve):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "setting": setting,
                    "schedule": schedule,
                    "alpha": alpha,
                    "episode": episode,
                    "labels": labels_for(dataset, episode),
                    "accuracy": accuracy,
                    "run": str(run_dir),
                    "mtime": run_dir.stat().st_mtime,
                }
            )
    return newest_unique(rows)


def collect_alpha0_baselines() -> pd.DataFrame:
    rows: list[dict] = []
    for run_dir in OUTPUT_ROOT.glob("CIFAR*/resnet18/**/maxherding_dnorm*alpha0_weighted*"):
        if not run_dir.is_dir():
            continue
        match = BASELINE_RE.search(run_dir.name)
        if not match:
            continue
        dataset = dataset_name(match.group("dataset"))
        seed = int(match.group("seed"))
        setting = setting_label(match)
        curve = load_curve(run_dir)
        for episode, accuracy in enumerate(curve):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "setting": setting,
                    "schedule": "alpha0_baseline",
                    "alpha": 0.0,
                    "episode": episode,
                    "labels": labels_for(dataset, episode),
                    "accuracy": accuracy,
                    "run": str(run_dir),
                    "mtime": run_dir.stat().st_mtime,
                }
            )
    return newest_unique(rows)


def add_baseline_delta(decay: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    base = baseline[
        ["dataset", "seed", "setting", "episode", "accuracy"]
    ].rename(columns={"accuracy": "baseline_accuracy"})
    out = decay.merge(base, on=["dataset", "seed", "setting", "episode"], how="left")
    out["delta_vs_alpha0"] = out["accuracy"] - out["baseline_accuracy"]
    return out


def summarize(decay: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, pd.DataFrame]:
    delta = add_baseline_delta(decay, baseline)
    by_episode_mean = (
        delta.groupby(["dataset", "setting", "schedule", "alpha", "episode", "labels"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_delta_vs_alpha0=("delta_vs_alpha0", "mean"),
            std_delta_vs_alpha0=("delta_vs_alpha0", "std"),
            seeds=("seed", "nunique"),
        )
        .sort_values(["dataset", "setting", "schedule", "alpha", "episode"])
    )
    final = (
        delta.sort_values("episode")
        .groupby(["dataset", "seed", "setting", "schedule", "alpha"], as_index=False)
        .tail(1)
        .sort_values(["dataset", "setting", "schedule", "alpha", "seed"])
    )
    final_mean = (
        final.groupby(["dataset", "setting", "schedule", "alpha"], as_index=False)
        .agg(
            mean_final_accuracy=("accuracy", "mean"),
            std_final_accuracy=("accuracy", "std"),
            mean_final_delta_vs_alpha0=("delta_vs_alpha0", "mean"),
            std_final_delta_vs_alpha0=("delta_vs_alpha0", "std"),
            seeds=("seed", "nunique"),
        )
        .sort_values(["dataset", "setting", "mean_final_accuracy"], ascending=[True, True, False])
    )
    best_by_budget = (
        by_episode_mean.sort_values(
            ["dataset", "setting", "episode", "mean_delta_vs_alpha0"],
            ascending=[True, True, True, False],
        )
        .groupby(["dataset", "setting", "episode"], as_index=False)
        .head(1)
        .sort_values(["dataset", "setting", "episode"])
    )
    phase = by_episode_mean.copy()
    phase["phase"] = pd.cut(
        phase["episode"],
        bins=[-1, 2, 5, 9],
        labels=["early", "mid", "late"],
    )
    phase_summary = (
        phase.groupby(["dataset", "setting", "schedule", "alpha", "phase"], observed=True)
        .agg(
            mean_phase_accuracy=("mean_accuracy", "mean"),
            mean_phase_delta_vs_alpha0=("mean_delta_vs_alpha0", "mean"),
            seeds=("seeds", "max"),
        )
        .reset_index()
        .sort_values(["dataset", "setting", "schedule", "alpha", "phase"])
    )
    return {
        "by_episode": delta,
        "by_episode_mean": by_episode_mean,
        "best_by_budget": best_by_budget,
        "phase_summary": phase_summary,
        "final_by_seed": final,
        "final_mean": final_mean,
    }


def plot_curves(decay_mean: pd.DataFrame, baseline: pd.DataFrame) -> None:
    baseline_mean = (
        baseline.groupby(["dataset", "setting", "episode", "labels"], as_index=False)
        .agg(mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"))
    )
    for dataset in sorted(decay_mean["dataset"].unique()):
        for setting in sorted(decay_mean[decay_mean["dataset"] == dataset]["setting"].unique()):
            part = decay_mean[(decay_mean["dataset"] == dataset) & (decay_mean["setting"] == setting)]
            base = baseline_mean[
                (baseline_mean["dataset"] == dataset) & (baseline_mean["setting"] == setting)
            ].sort_values("labels")

            fig, ax = plt.subplots(figsize=(9, 5))
            if not base.empty:
                ax.plot(
                    base["labels"],
                    base["mean_accuracy"],
                    color="black",
                    linewidth=2.2,
                    marker="o",
                    label="alpha=0 baseline",
                )
            for (schedule, alpha), group in part.groupby(["schedule", "alpha"]):
                group = group.sort_values("labels")
                ax.plot(
                    group["labels"],
                    group["mean_accuracy"],
                    marker="o",
                    linewidth=1.7,
                    label=f"{schedule}, alpha={alpha:g}",
                )
            ax.set_title(f"{dataset} {setting}: tuned density-normalization schedules")
            ax.set_xlabel("Labeled points")
            ax.set_ylabel("Accuracy")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / f"{dataset.lower()}_{setting.replace('=', '')}_accuracy.png", dpi=160)
            plt.close(fig)


def plot_deltas(decay_mean: pd.DataFrame) -> None:
    for dataset in sorted(decay_mean["dataset"].unique()):
        for setting in sorted(decay_mean[decay_mean["dataset"] == dataset]["setting"].unique()):
            part = decay_mean[(decay_mean["dataset"] == dataset) & (decay_mean["setting"] == setting)]
            fig, ax = plt.subplots(figsize=(9, 5))
            for (schedule, alpha), group in part.groupby(["schedule", "alpha"]):
                group = group.sort_values("labels")
                ax.plot(
                    group["labels"],
                    group["mean_delta_vs_alpha0"],
                    marker="o",
                    linewidth=1.7,
                    label=f"{schedule}, alpha={alpha:g}",
                )
            ax.axhline(0, color="black", linewidth=1)
            ax.set_title(f"{dataset} {setting}: schedule delta vs alpha=0")
            ax.set_xlabel("Labeled points")
            ax.set_ylabel("Accuracy delta")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / f"{dataset.lower()}_{setting.replace('=', '')}_delta.png", dpi=160)
            plt.close(fig)


def plot_final(final_mean: pd.DataFrame) -> None:
    for dataset in sorted(final_mean["dataset"].unique()):
        part = final_mean[final_mean["dataset"] == dataset].copy()
        part["label"] = part["setting"] + "\n" + part["schedule"] + f"\nalpha=" + part["alpha"].map("{:g}".format)
        fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(part)), 5))
        colors = ["#4C78A8" if s == "global" else "#F58518" for s in part["setting"]]
        ax.bar(range(len(part)), part["mean_final_delta_vs_alpha0"], color=colors)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(range(len(part)))
        ax.set_xticklabels(part["label"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Final accuracy delta vs alpha=0")
        ax.set_title(f"{dataset}: final tuned-schedule gain/loss")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{dataset.lower()}_final_delta_bars.png", dpi=160)
        plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    decay = collect_decay_runs()
    baseline = collect_alpha0_baselines()

    decay.to_csv(RESULTS_DIR / "decay_runs_by_episode.csv", index=False)
    baseline.to_csv(RESULTS_DIR / "alpha0_baselines_by_episode.csv", index=False)

    if decay.empty:
        print("No tuned decay runs found.")
        print(f"Checked under: {OUTPUT_ROOT.resolve()}")
        print(f"Wrote empty run table to: {RESULTS_DIR / 'decay_runs_by_episode.csv'}")
        return
    if baseline.empty:
        raise RuntimeError("Found decay runs but no alpha=0 weighted baselines.")

    tables = summarize(decay, baseline)
    for name, table in tables.items():
        table.to_csv(RESULTS_DIR / f"{name}.csv", index=False)

    plot_curves(tables["by_episode_mean"], baseline)
    plot_deltas(tables["by_episode_mean"])
    plot_final(tables["final_mean"])

    print(f"Found {decay[['dataset', 'seed', 'setting', 'schedule', 'alpha']].drop_duplicates().shape[0]} decay runs.")
    print(f"Wrote CSVs to {RESULTS_DIR}")
    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
