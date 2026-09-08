#!/usr/bin/env python3
"""Generate selected-density comparisons for MaxHerding local sigma runs.

This joins selected-point density from ``selection_episode_summary.csv`` with
dataset-level density from ``density_summary.csv``. The important normalized
quantities are:

``selected_raw_to_dataset_raw_ratio``
    Selected-point raw/global degree divided by the average raw/global dataset
    degree for the same run.

``selected_effective_to_dataset_effective_ratio``
    Selected-point effective degree divided by the average effective dataset
    degree for the same run. For local sigma, this uses the local graph induced
    by that K.

The script also compares every local-K run against the matching global run for
the same dataset, seed, and episode.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_DIR = Path("results_summaries/maxherding_density_selection")
OUTPUT_DIR = INPUT_DIR / "comparisons"
PLOTS_DIR = Path("visual_outputs/maxherding/density_selection/comparisons")


def setting_from_row(row: pd.Series) -> str:
    if not bool(row["local_sigma"]):
        return "global"
    return f"k={int(row['local_k'])}"


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace({0: pd.NA})


def load_joined() -> pd.DataFrame:
    sel = pd.read_csv(INPUT_DIR / "selection_episode_summary.csv")
    density = pd.read_csv(INPUT_DIR / "density_summary.csv")

    sel["setting"] = sel.apply(setting_from_row, axis=1)
    density["setting"] = density.apply(setting_from_row, axis=1)

    density_cols = [
        "run_name",
        "dataset",
        "seed",
        "setting",
        "raw_degree_mean",
        "raw_degree_median",
        "raw_degree_std",
        "effective_degree_mean",
        "effective_degree_median",
        "effective_degree_std",
    ]
    density = density[density_cols].rename(
        columns={
            "raw_degree_mean": "dataset_raw_degree_mean",
            "raw_degree_median": "dataset_raw_degree_median",
            "raw_degree_std": "dataset_raw_degree_std",
            "effective_degree_mean": "dataset_effective_degree_mean",
            "effective_degree_median": "dataset_effective_degree_median",
            "effective_degree_std": "dataset_effective_degree_std",
        }
    )

    keep_cols = [
        "run_name",
        "run_dir",
        "dataset",
        "seed",
        "episode",
        "labels_after_episode",
        "setting",
        "local_sigma",
        "local_k",
        "test_accuracy_after_episode",
        "selected_raw_degree_mean",
        "selected_effective_degree_mean",
        "selected_sigma_mean",
        "unique_selected_classes",
        "selected_label_entropy_normalized",
    ]
    joined = sel[keep_cols].merge(
        density,
        on=["run_name", "dataset", "seed", "setting"],
        how="left",
    )
    joined["selected_raw_to_dataset_raw_ratio"] = safe_ratio(
        joined["selected_raw_degree_mean"],
        joined["dataset_raw_degree_mean"],
    )
    joined["selected_effective_to_dataset_effective_ratio"] = safe_ratio(
        joined["selected_effective_degree_mean"],
        joined["dataset_effective_degree_mean"],
    )
    return joined


def local_vs_global(joined: pd.DataFrame) -> pd.DataFrame:
    global_rows = joined[joined["setting"] == "global"][
        [
            "dataset",
            "seed",
            "episode",
            "selected_raw_degree_mean",
            "selected_effective_degree_mean",
            "dataset_raw_degree_mean",
            "dataset_effective_degree_mean",
            "selected_raw_to_dataset_raw_ratio",
            "selected_effective_to_dataset_effective_ratio",
            "test_accuracy_after_episode",
            "unique_selected_classes",
            "selected_label_entropy_normalized",
        ]
    ].rename(
        columns={
            "selected_raw_degree_mean": "global_selected_raw_degree_mean",
            "selected_effective_degree_mean": "global_selected_effective_degree_mean",
            "dataset_raw_degree_mean": "global_dataset_raw_degree_mean",
            "dataset_effective_degree_mean": "global_dataset_effective_degree_mean",
            "selected_raw_to_dataset_raw_ratio": "global_selected_raw_to_dataset_raw_ratio",
            "selected_effective_to_dataset_effective_ratio": "global_selected_effective_to_dataset_effective_ratio",
            "test_accuracy_after_episode": "global_accuracy",
            "unique_selected_classes": "global_unique_selected_classes",
            "selected_label_entropy_normalized": "global_entropy_norm",
        }
    )

    local = joined[joined["setting"] != "global"].copy()
    cmp = local.merge(global_rows, on=["dataset", "seed", "episode"], how="left")

    cmp["raw_degree_delta_vs_global"] = (
        cmp["selected_raw_degree_mean"] - cmp["global_selected_raw_degree_mean"]
    )
    cmp["raw_degree_ratio_vs_global"] = safe_ratio(
        cmp["selected_raw_degree_mean"],
        cmp["global_selected_raw_degree_mean"],
    )
    cmp["effective_degree_delta_vs_global"] = (
        cmp["selected_effective_degree_mean"]
        - cmp["global_selected_effective_degree_mean"]
    )
    cmp["effective_degree_ratio_vs_global"] = safe_ratio(
        cmp["selected_effective_degree_mean"],
        cmp["global_selected_effective_degree_mean"],
    )
    cmp["selected_raw_to_dataset_ratio_delta_vs_global"] = (
        cmp["selected_raw_to_dataset_raw_ratio"]
        - cmp["global_selected_raw_to_dataset_raw_ratio"]
    )
    cmp["selected_raw_to_dataset_ratio_ratio_vs_global"] = safe_ratio(
        cmp["selected_raw_to_dataset_raw_ratio"],
        cmp["global_selected_raw_to_dataset_raw_ratio"],
    )
    cmp["selected_effective_to_dataset_ratio_delta_vs_global"] = (
        cmp["selected_effective_to_dataset_effective_ratio"]
        - cmp["global_selected_effective_to_dataset_effective_ratio"]
    )
    cmp["selected_effective_to_dataset_ratio_ratio_vs_global"] = safe_ratio(
        cmp["selected_effective_to_dataset_effective_ratio"],
        cmp["global_selected_effective_to_dataset_effective_ratio"],
    )
    cmp["accuracy_delta_vs_global"] = (
        cmp["test_accuracy_after_episode"] - cmp["global_accuracy"]
    )
    cmp["unique_classes_delta_vs_global"] = (
        cmp["unique_selected_classes"] - cmp["global_unique_selected_classes"]
    )
    cmp["entropy_delta_vs_global"] = (
        cmp["selected_label_entropy_normalized"] - cmp["global_entropy_norm"]
    )
    return cmp


def summarize(cmp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg_kwargs = dict(
        selected_raw_degree_mean=("selected_raw_degree_mean", "mean"),
        dataset_raw_degree_mean=("dataset_raw_degree_mean", "mean"),
        selected_raw_to_dataset_raw_ratio=("selected_raw_to_dataset_raw_ratio", "mean"),
        global_selected_raw_to_dataset_raw_ratio=(
            "global_selected_raw_to_dataset_raw_ratio",
            "mean",
        ),
        selected_raw_to_dataset_ratio_delta_vs_global=(
            "selected_raw_to_dataset_ratio_delta_vs_global",
            "mean",
        ),
        selected_raw_to_dataset_ratio_ratio_vs_global=(
            "selected_raw_to_dataset_ratio_ratio_vs_global",
            "mean",
        ),
        selected_effective_degree_mean=("selected_effective_degree_mean", "mean"),
        dataset_effective_degree_mean=("dataset_effective_degree_mean", "mean"),
        selected_effective_to_dataset_effective_ratio=(
            "selected_effective_to_dataset_effective_ratio",
            "mean",
        ),
        global_selected_effective_to_dataset_effective_ratio=(
            "global_selected_effective_to_dataset_effective_ratio",
            "mean",
        ),
        selected_effective_to_dataset_ratio_delta_vs_global=(
            "selected_effective_to_dataset_ratio_delta_vs_global",
            "mean",
        ),
        selected_effective_to_dataset_ratio_ratio_vs_global=(
            "selected_effective_to_dataset_ratio_ratio_vs_global",
            "mean",
        ),
        raw_degree_delta_vs_global=("raw_degree_delta_vs_global", "mean"),
        effective_degree_delta_vs_global=("effective_degree_delta_vs_global", "mean"),
        accuracy_delta_vs_global=("accuracy_delta_vs_global", "mean"),
        entropy_delta_vs_global=("entropy_delta_vs_global", "mean"),
        seeds=("seed", "nunique"),
        episodes=("episode", "nunique"),
    )

    avg = cmp.groupby(["dataset", "setting"], as_index=False).agg(**agg_kwargs)

    max_episode = cmp.groupby("dataset")["episode"].transform("max")
    final = (
        cmp[cmp["episode"] == max_episode]
        .groupby(["dataset", "setting"], as_index=False)
        .agg(**{k: v for k, v in agg_kwargs.items() if k != "episodes"})
    )
    return avg, final


def sort_setting_key(setting: str) -> int:
    if setting == "global":
        return 0
    return int(setting.split("=")[1])


def add_global_rows(joined: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Add global rows so plots show the baseline next to local K values."""
    global_summary = (
        joined[joined["setting"] == "global"]
        .groupby(["dataset", "setting"], as_index=False)
        .agg(
            selected_raw_to_dataset_raw_ratio=(
                "selected_raw_to_dataset_raw_ratio",
                "mean",
            ),
            selected_effective_to_dataset_effective_ratio=(
                "selected_effective_to_dataset_effective_ratio",
                "mean",
            ),
            selected_raw_degree_mean=("selected_raw_degree_mean", "mean"),
            dataset_raw_degree_mean=("dataset_raw_degree_mean", "mean"),
            selected_effective_degree_mean=("selected_effective_degree_mean", "mean"),
            dataset_effective_degree_mean=("dataset_effective_degree_mean", "mean"),
            seeds=("seed", "nunique"),
            episodes=("episode", "nunique"),
        )
    )
    combined = pd.concat([global_summary, summary], ignore_index=True, sort=False)
    combined["setting_sort"] = combined["setting"].map(sort_setting_key)
    return combined.sort_values(["dataset", "setting_sort"])


def final_global_rows(joined: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    max_episode = joined.groupby("dataset")["episode"].transform("max")
    final_joined = joined[joined["episode"] == max_episode]
    return add_global_rows(final_joined, summary)


def plot_ratio_summary(df: pd.DataFrame, suffix: str, title_suffix: str) -> None:
    """Plot selected/dataset density ratios for raw and effective graphs."""
    for dataset, part in df.groupby("dataset"):
        part = part.sort_values("setting_sort")
        x_labels = part["setting"].tolist()
        x = range(len(x_labels))

        fig, ax = plt.subplots(figsize=(10, 5.2))
        ax.plot(
            x,
            part["selected_raw_to_dataset_raw_ratio"],
            marker="o",
            linewidth=2,
            label="raw/global density ratio",
        )
        ax.plot(
            x,
            part["selected_effective_to_dataset_effective_ratio"],
            marker="o",
            linewidth=2,
            label="effective/local density ratio",
        )
        ax.axhline(1.0, color="black", linewidth=1, linestyle="--", label="dataset average")
        ax.set_xticks(list(x))
        ax.set_xticklabels(x_labels, rotation=35, ha="right")
        ax.set_ylabel("Selected-point degree / dataset-average degree")
        ax.set_xlabel("Sigma setting")
        ax.set_title(f"{dataset}: selected-point density normalized by dataset density ({title_suffix})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{dataset.lower()}_selected_density_ratio_{suffix}.png", dpi=170)
        plt.close(fig)

        # A second plot zooms into the raw/global-density ratio, which directly
        # answers whether local sigma selects points that are globally denser or
        # sparser than global MaxHerding selections.
        fig, ax = plt.subplots(figsize=(10, 5.2))
        ax.plot(
            x,
            part["selected_raw_to_dataset_raw_ratio"],
            marker="o",
            linewidth=2,
            color="#4C78A8",
        )
        ax.axhline(1.0, color="black", linewidth=1, linestyle="--", label="dataset average")
        global_val = float(part[part["setting"] == "global"]["selected_raw_to_dataset_raw_ratio"].iloc[0])
        ax.axhline(
            global_val,
            color="#F58518",
            linewidth=1.5,
            linestyle=":",
            label="global selected ratio",
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(x_labels, rotation=35, ha="right")
        ax.set_ylabel("Selected raw degree / dataset raw degree")
        ax.set_xlabel("Sigma setting")
        ax.set_title(f"{dataset}: raw/global selected density ratio ({title_suffix})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{dataset.lower()}_selected_raw_density_ratio_{suffix}.png", dpi=170)
        plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    joined = load_joined()
    cmp = local_vs_global(joined)
    avg, final = summarize(cmp)
    avg_with_global = add_global_rows(joined, avg)
    final_with_global = final_global_rows(joined, final)

    joined.to_csv(OUTPUT_DIR / "selected_density_by_run_episode.csv", index=False)
    cmp.to_csv(OUTPUT_DIR / "local_vs_global_selected_density_by_episode.csv", index=False)
    avg.to_csv(OUTPUT_DIR / "local_vs_global_selected_density_avg.csv", index=False)
    final.to_csv(OUTPUT_DIR / "local_vs_global_selected_density_final_episode.csv", index=False)
    avg_with_global.to_csv(OUTPUT_DIR / "selected_density_ratio_avg_with_global.csv", index=False)
    final_with_global.to_csv(OUTPUT_DIR / "selected_density_ratio_final_with_global.csv", index=False)

    plot_ratio_summary(avg_with_global, "avg", "avg over episodes/seeds")
    plot_ratio_summary(final_with_global, "final", "final episode")

    print(f"Wrote comparison tables to {OUTPUT_DIR}")
    print(f"Wrote comparison plots to {PLOTS_DIR}")
    print("\nAverage over episodes/seeds:")
    print(avg.to_string(index=False))
    print("\nFinal episode only:")
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
