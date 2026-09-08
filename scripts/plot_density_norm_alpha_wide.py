#!/usr/bin/env python3
"""Plot the weighted wide-alpha density-normalization sweep.

This script uses the canonical summary CSVs under
``results_summaries/maxherding_density_norm/alpha_wide`` and writes a graph set
that includes every tested alpha. The older pilot plots in
``visual_outputs/maxherding/density_norm`` only contain alpha 0.25 and 0.5.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results_summaries/maxherding_density_norm/alpha_wide")
PLOTS_DIR = Path("visual_outputs/maxherding/density_norm/wide_alpha_summary")


def _safe_setting_name(setting: str) -> str:
    return setting.replace("=", "")


def _plot_accuracy_curves(df: pd.DataFrame) -> None:
    for dataset in sorted(df["dataset"].unique()):
        for setting in sorted(df[df["dataset"] == dataset]["setting"].unique()):
            part = df[(df["dataset"] == dataset) & (df["setting"] == setting)]
            fig, ax = plt.subplots(figsize=(8.5, 5.0))
            for alpha, group in part.groupby("alpha"):
                group = group.sort_values("labels")
                ax.plot(
                    group["labels"],
                    group["accuracy"],
                    marker="o",
                    linewidth=1.7,
                    label=f"alpha={alpha:g}",
                )
            ax.set_title(f"{dataset} {setting}: all alpha values")
            ax.set_xlabel("Labeled points")
            ax.set_ylabel("Accuracy")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            out = PLOTS_DIR / (
                f"{dataset.lower()}_{_safe_setting_name(setting)}_all_alphas_accuracy.png"
            )
            fig.savefig(out, dpi=160)
            plt.close(fig)


def _plot_delta_curves(delta: pd.DataFrame) -> None:
    for dataset in sorted(delta["dataset"].unique()):
        for setting in sorted(delta[delta["dataset"] == dataset]["setting"].unique()):
            part = delta[
                (delta["dataset"] == dataset)
                & (delta["setting"] == setting)
                & (delta["alpha"] > 0)
            ]
            fig, ax = plt.subplots(figsize=(8.5, 5.0))
            for alpha, group in part.groupby("alpha"):
                group = group.sort_values("labels")
                ax.plot(
                    group["labels"],
                    group["delta_vs_alpha0"],
                    marker="o",
                    linewidth=1.7,
                    label=f"alpha={alpha:g}",
                )
            ax.axhline(0, color="black", linewidth=1)
            ax.set_title(f"{dataset} {setting}: accuracy delta vs alpha=0")
            ax.set_xlabel("Labeled points")
            ax.set_ylabel("Accuracy delta")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            out = PLOTS_DIR / (
                f"{dataset.lower()}_{_safe_setting_name(setting)}_all_alphas_delta.png"
            )
            fig.savefig(out, dpi=160)
            plt.close(fig)


def _plot_final_accuracy(final: pd.DataFrame) -> None:
    for dataset in sorted(final["dataset"].unique()):
        part = final[final["dataset"] == dataset]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for setting, group in part.groupby("setting"):
            group = group.sort_values("alpha")
            ax.plot(group["alpha"], group["accuracy"], marker="o", label=setting)
        ax.set_title(f"{dataset}: final accuracy vs alpha")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Final accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{dataset.lower()}_final_accuracy_vs_alpha.png", dpi=160)
        plt.close(fig)


def _plot_best_alpha_by_budget(best: pd.DataFrame) -> None:
    for dataset in sorted(best["dataset"].unique()):
        for setting in sorted(best[best["dataset"] == dataset]["setting"].unique()):
            part = best[
                (best["dataset"] == dataset) & (best["setting"] == setting)
            ].sort_values("labels")
            fig, ax = plt.subplots(figsize=(8.5, 4.6))
            ax.bar(part["labels"].astype(str), part["alpha"], color="#4C78A8")
            ax.set_title(f"{dataset} {setting}: best alpha by budget")
            ax.set_xlabel("Labeled points")
            ax.set_ylabel("Best alpha")
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            out = PLOTS_DIR / (
                f"{dataset.lower()}_{_safe_setting_name(setting)}_best_alpha_by_budget.png"
            )
            fig.savefig(out, dpi=160)
            plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    by_episode = pd.read_csv(RESULTS_DIR / "by_episode.csv")
    delta = pd.read_csv(RESULTS_DIR / "delta_vs_alpha0.csv")
    final = pd.read_csv(RESULTS_DIR / "final_accuracy.csv")
    best = pd.read_csv(RESULTS_DIR / "best_alpha_by_episode.csv")

    _plot_accuracy_curves(by_episode)
    _plot_delta_curves(delta)
    _plot_final_accuracy(final)
    _plot_best_alpha_by_budget(best)

    print(f"Wrote wide-alpha plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
