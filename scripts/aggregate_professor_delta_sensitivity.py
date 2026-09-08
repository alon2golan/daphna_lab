#!/usr/bin/env python3
"""Aggregate the professor-notes delta-sensitivity MaxHerding runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path("/cs/labs/daphna/ido.dotan/projects/daphna_lab")
RESULTS_DIR = ROOT / "results_summaries" / "professor_notes_2026_05_28"
PLOTS_DIR = ROOT / "visual_outputs" / "professor_notes_2026_05_28" / "delta_sensitivity"


def _cfg_get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _setting(cfg: dict[str, Any]) -> str:
    local = bool(_cfg_get(cfg, "LOCAL_SIGMA", _cfg_get(cfg, "ACTIVE_LEARNING.LOCAL_SIGMA", False)))
    if not local:
        return "global"
    k = _cfg_get(cfg, "LOCAL_SIGMA_K", _cfg_get(cfg, "ACTIVE_LEARNING.LOCAL_SIGMA_K"))
    return f"k={int(k)}"


def discover_runs() -> list[Path]:
    runs = []
    for run_dir in ROOT.glob("output/*/resnet18/**/profnotes_delta_*"):
        if (run_dir / "config.yaml").exists() and (run_dir / "plot_episode_yvalues.npy").exists():
            runs.append(run_dir)
    return sorted(runs)


def aggregate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for run_dir in discover_runs():
        with (run_dir / "config.yaml").open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        y_values = np.load(run_dir / "plot_episode_yvalues.npy")
        dataset = _cfg_get(cfg, "DATASET.NAME")
        seed = int(_cfg_get(cfg, "RNG_SEED"))
        budget_size = int(_cfg_get(cfg, "ACTIVE_LEARNING.BUDGET_SIZE"))
        delta = float(_cfg_get(cfg, "ACTIVE_LEARNING.INITIAL_DELTA"))
        alpha = float(_cfg_get(cfg, "DENSITY_NORM_ALPHA", _cfg_get(cfg, "ACTIVE_LEARNING.DENSITY_NORM_ALPHA", 0.0)))
        setting = _setting(cfg)
        for episode, y in enumerate(y_values):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "setting": setting,
                    "delta": delta,
                    "alpha": alpha,
                    "episode": episode,
                    "labels": int((episode + 1) * budget_size),
                    "accuracy": float(y),
                    "run_dir": str(run_dir),
                }
            )
    by_episode = pd.DataFrame(rows).sort_values(["dataset", "setting", "delta", "alpha", "seed", "labels"])
    by_episode.to_csv(RESULTS_DIR / "delta_sensitivity_by_episode.csv", index=False)

    final = (
        by_episode.sort_values("labels")
        .groupby(["dataset", "seed", "setting", "delta", "alpha"], as_index=False)
        .tail(1)
        .sort_values(["dataset", "setting", "delta", "alpha", "seed"])
    )
    final.to_csv(RESULTS_DIR / "delta_sensitivity_final_by_seed.csv", index=False)

    mean = (
        by_episode.groupby(["dataset", "setting", "delta", "alpha", "labels"], as_index=False)
        .agg(mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"), seeds=("seed", "nunique"))
        .sort_values(["dataset", "setting", "delta", "alpha", "labels"])
    )
    mean.to_csv(RESULTS_DIR / "delta_sensitivity_mean_by_episode.csv", index=False)

    baseline = mean[mean["alpha"] == 0][
        ["dataset", "setting", "delta", "labels", "mean_accuracy"]
    ].rename(columns={"mean_accuracy": "alpha0_accuracy"})
    gain = mean[mean["alpha"] != 0].merge(baseline, on=["dataset", "setting", "delta", "labels"], how="left")
    gain["gain_vs_alpha0"] = gain["mean_accuracy"] - gain["alpha0_accuracy"]
    gain.to_csv(RESULTS_DIR / "delta_sensitivity_gain_vs_alpha0.csv", index=False)

    return by_episode, mean, gain


def plot(mean: pd.DataFrame, gain: pd.DataFrame) -> list[Path]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for dataset in sorted(mean["dataset"].unique()):
        for setting in sorted(mean[mean["dataset"] == dataset]["setting"].unique()):
            part = gain[(gain["dataset"] == dataset) & (gain["setting"] == setting)]
            if not part.empty:
                fig, ax = plt.subplots(figsize=(9.2, 5.3))
                for delta, group in part.groupby("delta"):
                    group = group.sort_values("labels")
                    ax.plot(group["labels"], group["gain_vs_alpha0"], marker="o", linewidth=1.8, label=f"delta={delta:g}")
                ax.axhline(0, color="black", linestyle="--", linewidth=1.1)
                ax.set_title(f"{dataset} {setting}: alpha gain sensitivity to delta")
                ax.set_xlabel("Labeled points")
                ax.set_ylabel("Accuracy gain over alpha=0")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                out = PLOTS_DIR / f"{dataset.lower()}_{setting.replace('=', '')}_gain_vs_delta_by_budget.png"
                fig.tight_layout()
                fig.savefig(out)
                plt.close(fig)
                paths.append(out)

            final = (
                mean[(mean["dataset"] == dataset) & (mean["setting"] == setting)]
                .sort_values("labels")
                .groupby(["delta", "alpha"], as_index=False)
                .tail(1)
                .sort_values(["alpha", "delta"])
            )
            if not final.empty:
                fig, ax = plt.subplots(figsize=(8.2, 5.0))
                for alpha, group in final.groupby("alpha"):
                    ax.plot(group["delta"], group["mean_accuracy"], marker="o", linewidth=2.0, label=f"alpha={alpha:g}")
                ax.set_title(f"{dataset} {setting}: final accuracy vs delta")
                ax.set_xlabel("Delta")
                ax.set_ylabel("Final test accuracy (%)")
                ax.grid(True, alpha=0.3)
                ax.legend()
                out = PLOTS_DIR / f"{dataset.lower()}_{setting.replace('=', '')}_final_accuracy_vs_delta.png"
                fig.tight_layout()
                fig.savefig(out)
                plt.close(fig)
                paths.append(out)
    return paths


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _, mean, gain = aggregate()
    paths = plot(mean, gain)
    print(f"Wrote delta sensitivity CSVs under {RESULTS_DIR}")
    print(f"Wrote delta sensitivity plots under {PLOTS_DIR}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
