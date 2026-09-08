#!/usr/bin/env python3
"""Aggregate CIFAR100 controlled-degree MaxHerding runs."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_GLOB = "output/CIFAR100/resnet18/2026_6_27/maxherding_degree_control_cifar100_*"
OUTDIR = Path("results_summaries/cifar100_degree_controlled_2026_06_27")
PLOTS_DIR = Path("visual_outputs/cifar100_degree_controlled_2026_06_27")

RUN_RE = re.compile(
    r"maxherding_degree_control_cifar100_seed(?P<seed>\d+)_"
    r"(?P<mode>global|local)_k(?P<k>\d+)_delta(?P<delta>\d+p\d+)_"
    r"(?P<band>low|optimal|high)_deg(?P<degree>\d+p\d+)_"
)


def parse_run_name(run_dir: Path) -> dict[str, object]:
    match = RUN_RE.search(run_dir.name)
    if not match:
        raise ValueError(f"Cannot parse run name: {run_dir.name}")
    info = match.groupdict()
    delta = float(info["delta"].replace("p", "."))
    estimated_degree = float(info["degree"].replace("p", "."))
    k = int(info["k"])
    setting = "global" if info["mode"] == "global" else f"K={k}"
    config = f"{setting}, delta={delta:.2f}, degree~{estimated_degree:.1f}"
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "seed": int(info["seed"]),
        "mode": info["mode"],
        "local_k": k if info["mode"] == "local" else np.nan,
        "setting": setting,
        "delta": delta,
        "target_band": info["band"],
        "estimated_effective_degree": estimated_degree,
        "config": config,
    }


def load_runs() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(Path(".").glob(RUN_GLOB)):
        x_path = run_dir / "plot_episode_xvalues.npy"
        y_path = run_dir / "plot_episode_yvalues.npy"
        if not x_path.exists() or not y_path.exists():
            continue
        info = parse_run_name(run_dir)
        episodes = np.load(x_path).astype(int)
        accs = np.load(y_path).astype(float)
        for episode, accuracy in zip(episodes, accs):
            row = dict(info)
            row.update(
                {
                    "episode": int(episode),
                    "budget_labels": int((episode + 1) * 100),
                    "accuracy": float(accuracy),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by_config = (
        df.groupby(
            [
                "target_band",
                "mode",
                "setting",
                "local_k",
                "delta",
                "estimated_effective_degree",
                "config",
                "budget_labels",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"), seeds=("seed", "nunique"))
        .sort_values(["budget_labels", "target_band", "estimated_effective_degree", "config"])
    )
    final_by_config = by_config[by_config["budget_labels"] == by_config["budget_labels"].max()].copy()
    final_by_config = final_by_config.sort_values("mean_accuracy", ascending=False)

    by_band = (
        df.groupby(["target_band", "budget_labels"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            configs=("config", "nunique"),
            seeds=("seed", "nunique"),
        )
        .sort_values(["budget_labels", "target_band"])
    )
    final_by_band = by_band[by_band["budget_labels"] == by_band["budget_labels"].max()].copy()
    final_by_band = final_by_band.sort_values("mean_accuracy", ascending=False)

    per_seed_final = df[df["budget_labels"] == df["budget_labels"].max()].copy()

    band_spread = (
        final_by_config.groupby("target_band", as_index=False)
        .agg(
            final_mean_across_configs=("mean_accuracy", "mean"),
            final_std_across_configs=("mean_accuracy", "std"),
            final_min_config=("mean_accuracy", "min"),
            final_max_config=("mean_accuracy", "max"),
            configs=("config", "nunique"),
        )
        .sort_values("final_mean_across_configs", ascending=False)
    )

    budget_best = (
        by_config.sort_values(["budget_labels", "mean_accuracy"], ascending=[True, False])
        .groupby("budget_labels", as_index=False)
        .head(1)
        .sort_values("budget_labels")
    )

    corr_rows = []
    for budget, group in by_config.groupby("budget_labels"):
        corr_rows.append(
            {
                "budget_labels": budget,
                "pearson_accuracy_vs_estimated_degree": group["mean_accuracy"].corr(
                    group["estimated_effective_degree"], method="pearson"
                ),
                "spearman_accuracy_vs_estimated_degree": group["mean_accuracy"].corr(
                    group["estimated_effective_degree"], method="spearman"
                ),
            }
        )
    correlations = pd.DataFrame(corr_rows)

    return {
        "by_config": by_config,
        "final_by_config": final_by_config,
        "by_band": by_band,
        "final_by_band": final_by_band,
        "per_seed_final": per_seed_final,
        "band_spread": band_spread,
        "budget_best": budget_best,
        "correlations": correlations,
    }


def plot_results(tables: dict[str, pd.DataFrame]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    by_band = tables["by_band"]
    by_config = tables["by_config"]
    final_by_config = tables["final_by_config"]
    budget_best = tables["budget_best"]

    colors = {"low": "#1f77b4", "optimal": "#2ca02c", "high": "#d62728"}

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for band, group in by_band.groupby("target_band"):
        group = group.sort_values("budget_labels")
        ax.errorbar(
            group["budget_labels"],
            group["mean_accuracy"],
            yerr=group["std_accuracy"],
            marker="o",
            capsize=3,
            color=colors.get(band),
            label=band,
        )
    ax.set_xlabel("Labeled points")
    ax.set_ylabel("Accuracy")
    ax.set_title("CIFAR100 Accuracy by Target Degree Band")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Degree band", loc="best")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cifar100_degree_controlled_band_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    final_plot = final_by_config.sort_values("estimated_effective_degree")
    for band, group in final_plot.groupby("target_band"):
        ax.errorbar(
            group["estimated_effective_degree"],
            group["mean_accuracy"],
            yerr=group["std_accuracy"],
            fmt="o",
            capsize=3,
            color=colors.get(band),
            label=band,
        )
        for row in group.itertuples(index=False):
            ax.annotate(
                f"{row.setting}\nδ={row.delta:.2f}",
                (row.estimated_effective_degree, row.mean_accuracy),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
    ax.set_xlabel("Estimated effective degree")
    ax.set_ylabel("Final accuracy at 1000 labels")
    ax.set_title("CIFAR100 Final Accuracy vs Estimated Effective Degree")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Degree band", loc="best")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cifar100_degree_controlled_final_accuracy_vs_degree.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    order = final_by_config.sort_values("mean_accuracy", ascending=True)
    labels = [
        f"{row.target_band}: {row.setting}, δ={row.delta:.2f}, deg~{row.estimated_effective_degree:.1f}"
        for row in order.itertuples(index=False)
    ]
    ax.barh(labels, order["mean_accuracy"], xerr=order["std_accuracy"], color=[colors[x] for x in order["target_band"]])
    ax.set_xlabel("Final accuracy at 1000 labels")
    ax.set_title("CIFAR100 Final Accuracy by Controlled-Degree Configuration")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cifar100_degree_controlled_final_accuracy_by_config.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    for config, group in by_config.groupby("config"):
        group = group.sort_values("budget_labels")
        band = group["target_band"].iloc[0]
        ax.plot(
            group["budget_labels"],
            group["mean_accuracy"],
            marker="o",
            alpha=0.65,
            color=colors.get(band),
            linewidth=1.5,
            label=config,
        )
    ax.set_xlabel("Labeled points")
    ax.set_ylabel("Accuracy")
    ax.set_title("CIFAR100 Accuracy Curves by Controlled-Degree Configuration")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Configuration", loc="lower right", fontsize=6, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cifar100_degree_controlled_all_config_curves.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), sharey=True)
    band_order = ["low", "optimal", "high"]
    for ax, band in zip(axes, band_order):
        band_df = by_config[by_config["target_band"] == band]
        for config, group in band_df.groupby("config"):
            group = group.sort_values("budget_labels")
            ax.plot(
                group["budget_labels"],
                group["mean_accuracy"],
                marker="o",
                linewidth=1.5,
                label=config,
            )
        ax.set_title(f"{band.capitalize()} degree")
        ax.set_xlabel("Labeled points")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc="lower right", framealpha=0.9)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle("CIFAR100 Controlled-Degree Accuracy Curves by Band")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cifar100_degree_controlled_band_panels.png")
    plt.close(fig)

    k1440 = by_config[
        (by_config["setting"] == "K=1440")
        & (by_config["delta"].isin([0.60, 0.65, 0.70]))
    ].copy()
    if not k1440.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        for _, cfg in (
            k1440[["target_band", "config", "estimated_effective_degree"]]
            .drop_duplicates()
            .sort_values("estimated_effective_degree")
            .iterrows()
        ):
            group = k1440[k1440["config"] == cfg["config"]].sort_values("budget_labels")
            label = f"{cfg['target_band']}: degree~{cfg['estimated_effective_degree']:.1f}"
            ax.plot(
                group["budget_labels"],
                group["mean_accuracy"],
                marker="o",
                linewidth=2.0,
                label=label,
            )
        ax.set_xlabel("Labeled points")
        ax.set_ylabel("Accuracy")
        ax.set_title("CIFAR100 Controlled Degree with Fixed K=1440")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "cifar100_degree_controlled_fixed_k1440_curves.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(
        budget_best["budget_labels"],
        budget_best["estimated_effective_degree"],
        marker="o",
        color="#9467bd",
    )
    for row in budget_best.itertuples(index=False):
        ax.annotate(
            f"{row.setting}, δ={row.delta:.2f}",
            (row.budget_labels, row.estimated_effective_degree),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("Labeled points")
    ax.set_ylabel("Estimated effective degree")
    ax.set_title("CIFAR100 Estimated Effective Degree of Best Configuration")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cifar100_degree_controlled_best_degree_per_budget.png")
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_runs()
    df.to_csv(OUTDIR / "cifar100_degree_controlled_by_seed_episode.csv", index=False)
    tables = summarize(df)
    for name, table in tables.items():
        table.to_csv(OUTDIR / f"cifar100_degree_controlled_{name}.csv", index=False)
    plot_results(tables)
    print(f"Loaded {df['run_name'].nunique()} runs")
    print(f"Wrote CSVs under {OUTDIR}")
    print(f"Wrote plots under {PLOTS_DIR}")


if __name__ == "__main__":
    main()
