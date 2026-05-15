#!/usr/bin/env python3
"""Create compact tables and plots for the MaxHerding density-normalization pilot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results_summaries/maxherding_density_norm")
PLOTS_DIR = Path("visual_outputs/maxherding/density_norm")


def setting_label(row: pd.Series) -> str:
    if row["sigma_config"] == "global":
        return "global"
    return f"k={int(row['k'])}"


def load_long() -> pd.DataFrame:
    runs = pd.read_csv(RESULTS_DIR / "density_norm_pilot_runs.csv")
    rows = []
    for _, run in runs.iterrows():
        ys = [float(y) for y in str(run["acc_curve"]).split(";")]
        # In these AL outputs, plot_episode_xvalues.npy stores episode indices
        # (0..9), not the number of labeled points. Reconstruct the actual label
        # budget from the dataset and episode so figures/tables use the real
        # x-axis: CIFAR10 10..100, CIFAR100 100..1000.
        budget = 10 if run["dataset"] == "CIFAR10" else 100
        for episode, accuracy in enumerate(ys):
            rows.append(
                {
                    "dataset": run["dataset"],
                    "setting": setting_label(run),
                    "degree_type": run["degree_type"],
                    "alpha": float(run["alpha"]),
                    "episode": episode,
                    "labels": budget * (episode + 1),
                    "accuracy": accuracy,
                    "final_acc": float(run["final_acc"]),
                }
            )
    return pd.DataFrame(rows)


def add_baseline_delta(long_df: pd.DataFrame) -> pd.DataFrame:
    baseline = long_df[long_df["alpha"] == 0][
        ["dataset", "setting", "degree_type", "episode", "accuracy"]
    ].rename(columns={"accuracy": "baseline_accuracy"})
    out = long_df.merge(
        baseline,
        on=["dataset", "setting", "degree_type", "episode"],
        how="left",
    )
    out["delta_vs_alpha0"] = out["accuracy"] - out["baseline_accuracy"]
    return out


def write_summary_tables(long_df: pd.DataFrame) -> None:
    final = (
        long_df.sort_values("episode")
        .groupby(["dataset", "setting", "degree_type", "alpha"], as_index=False)
        .tail(1)
        .sort_values(["dataset", "setting", "degree_type", "alpha"])
    )
    final.to_csv(RESULTS_DIR / "density_norm_pilot_final_summary.csv", index=False)

    delta = add_baseline_delta(long_df)
    delta.to_csv(RESULTS_DIR / "density_norm_pilot_by_episode.csv", index=False)

    delta["phase"] = pd.cut(
        delta["episode"],
        bins=[-1, 2, 5, 9],
        labels=["early", "mid", "late"],
    )
    phase = (
        delta[delta["alpha"] > 0]
        .groupby(["dataset", "setting", "degree_type", "alpha", "phase"], observed=True)[
            "delta_vs_alpha0"
        ]
        .mean()
        .reset_index()
    )
    phase.to_csv(RESULTS_DIR / "density_norm_pilot_phase_deltas.csv", index=False)


def plot_accuracy_curves(long_df: pd.DataFrame) -> None:
    for dataset in sorted(long_df["dataset"].unique()):
        for setting in sorted(long_df[long_df["dataset"] == dataset]["setting"].unique()):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
            for ax, degree_type in zip(axes, ["effective", "weighted"]):
                part = long_df[
                    (long_df["dataset"] == dataset)
                    & (long_df["setting"] == setting)
                    & (long_df["degree_type"] == degree_type)
                ]
                for alpha, group in part.groupby("alpha"):
                    group = group.sort_values("labels")
                    ax.plot(group["labels"], group["accuracy"], marker="o", label=f"alpha={alpha:g}")
                ax.set_title(f"{degree_type} degree")
                ax.set_xlabel("Labeled points")
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel("Accuracy")
            axes[1].legend(loc="best", fontsize=8)
            fig.suptitle(f"{dataset} {setting}: density-normalized MaxHerding")
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / f"{dataset.lower()}_{setting.replace('=', '')}_accuracy_curves.png", dpi=160)
            plt.close(fig)


def plot_delta_curves(long_df: pd.DataFrame) -> None:
    delta = add_baseline_delta(long_df)
    delta = delta[delta["alpha"] > 0]
    for dataset in sorted(delta["dataset"].unique()):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False, sharey=True)
        axes = axes.ravel()
        combos = list(
            delta[delta["dataset"] == dataset][["setting", "degree_type"]]
            .drop_duplicates()
            .sort_values(["setting", "degree_type"])
            .itertuples(index=False, name=None)
        )
        for ax, (setting, degree_type) in zip(axes, combos):
            part = delta[
                (delta["dataset"] == dataset)
                & (delta["setting"] == setting)
                & (delta["degree_type"] == degree_type)
            ]
            for alpha, group in part.groupby("alpha"):
                group = group.sort_values("labels")
                ax.plot(group["labels"], group["delta_vs_alpha0"], marker="o", label=f"alpha={alpha:g}")
            ax.axhline(0, color="black", linewidth=1)
            ax.set_title(f"{setting}, {degree_type}")
            ax.set_xlabel("Labeled points")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("Delta accuracy vs alpha=0")
        axes[2].set_ylabel("Delta accuracy vs alpha=0")
        axes[-1].legend(loc="best", fontsize=8)
        fig.suptitle(f"{dataset}: density-normalization gain/loss by budget")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{dataset.lower()}_delta_vs_alpha0_by_budget.png", dpi=160)
        plt.close(fig)


def plot_final_bars(long_df: pd.DataFrame) -> None:
    final = (
        long_df.sort_values("episode")
        .groupby(["dataset", "setting", "degree_type", "alpha"], as_index=False)
        .tail(1)
    )
    for dataset in sorted(final["dataset"].unique()):
        part = final[final["dataset"] == dataset].copy()
        part["config"] = part["setting"] + " / " + part["degree_type"]
        configs = sorted(part["config"].unique())
        fig, ax = plt.subplots(figsize=(11, 4.5))
        width = 0.22
        alphas = sorted(part["alpha"].unique())
        x_positions = range(len(configs))
        for offset_idx, alpha in enumerate(alphas):
            vals = []
            for config in configs:
                match = part[(part["config"] == config) & (part["alpha"] == alpha)]
                vals.append(float(match["accuracy"].iloc[0]) if not match.empty else float("nan"))
            offset = (offset_idx - (len(alphas) - 1) / 2) * width
            ax.bar([x + offset for x in x_positions], vals, width=width, label=f"alpha={alpha:g}")
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(configs, rotation=20, ha="right")
        ax.set_ylabel("Final accuracy")
        ax.set_title(f"{dataset}: final accuracy by normalization setting")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{dataset.lower()}_final_accuracy_bars.png", dpi=160)
        plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    long_df = load_long()
    write_summary_tables(long_df)
    plot_accuracy_curves(long_df)
    plot_delta_curves(long_df)
    plot_final_bars(long_df)
    print(f"Wrote plots to {PLOTS_DIR}")
    print(f"Wrote summary tables to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
