#!/usr/bin/env python3
"""Create Tiny ImageNet MaxHerding plots in the same style as the CIFAR plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/cs/labs/daphna/ido.dotan/projects/daphna_lab")
RESULTS_DIR = ROOT / "results_summaries" / "maxherding_tinyimagenet_maor_density_selection"
PLOTS_DIR = ROOT / "visual_outputs" / "tinyimagenet_maor_degree_study" / "cifar_style"


def _setting_sort_key(setting: str) -> tuple[int, int]:
    if setting == "global":
        return (0, 0)
    return (1, int(setting.split("=", 1)[1]))


def _setting_label(row: pd.Series) -> str:
    if row["setting"] == "global":
        return "global"
    return f"K={int(row['local_k'])}"


def _load_accuracy() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "accuracy_by_episode.csv")
    df["plot_setting"] = df.apply(_setting_label, axis=1)
    return df


def _load_summary() -> pd.DataFrame:
    summary = pd.read_csv(RESULTS_DIR / "density_selection_comparison_summary.csv")
    summary["plot_setting"] = summary["setting_name"]
    summary["plot_k"] = summary["local_k"].fillna(0).astype(int)
    return summary.sort_values("plot_setting", key=lambda s: s.map(_setting_sort_key))


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_all_k_curves(acc: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    settings = sorted(acc["plot_setting"].unique(), key=_setting_sort_key)
    local_settings = [s for s in settings if s != "global"]
    cmap = plt.get_cmap("viridis")

    global_df = acc[acc["plot_setting"] == "global"].sort_values("labels")
    ax.plot(
        global_df["labels"],
        global_df["accuracy"],
        color="#d62728",
        linewidth=2.8,
        marker="o",
        markersize=4.5,
        label="Global",
    )

    for idx, setting in enumerate(local_settings):
        group = acc[acc["plot_setting"] == setting].sort_values("labels")
        color = cmap(idx / max(1, len(local_settings) - 1))
        ax.plot(
            group["labels"],
            group["accuracy"],
            color=color,
            linewidth=1.9,
            marker="o",
            markersize=3.8,
            label=setting,
        )

    ymin = acc["accuracy"].min() - 0.8
    ymax = acc["accuracy"].max() + 0.8
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(sorted(acc["labels"].unique()))
    ax.set_xlabel("Number of labeled points")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Tiny ImageNet MaxHerding + 1NN: global sigma vs local K")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)

    out = PLOTS_DIR / "tinyimagenet_maor_all_k_curves.png"
    _finish(fig, out)
    return out


def plot_best_k_vs_global(acc: pd.DataFrame, summary: pd.DataFrame) -> Path:
    local_summary = summary[summary["plot_setting"] != "global"]
    best = local_summary.sort_values("final_accuracy", ascending=False).iloc[0]
    best_setting = best["plot_setting"]

    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    for setting, color, width in [
        ("global", "#d62728", 2.8),
        (best_setting, "#1f77b4", 2.8),
    ]:
        group = acc[acc["plot_setting"] == setting].sort_values("labels")
        label = "Global" if setting == "global" else f"Best local {setting}"
        ax.plot(
            group["labels"],
            group["accuracy"],
            color=color,
            linewidth=width,
            marker="o",
            markersize=4.5,
            label=label,
        )

    subset = acc[acc["plot_setting"].isin(["global", best_setting])]
    ax.set_ylim(subset["accuracy"].min() - 0.5, subset["accuracy"].max() + 0.5)
    ax.set_xticks(sorted(acc["labels"].unique()))
    ax.set_xlabel("Number of labeled points")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Tiny ImageNet MaxHerding + 1NN: best local K vs global")
    ax.grid(True, alpha=0.25)
    ax.legend()

    out = PLOTS_DIR / "tinyimagenet_maor_best_k_vs_global.png"
    _finish(fig, out)
    return out


def plot_final_accuracy_vs_k(summary: pd.DataFrame) -> Path:
    global_final = float(summary.loc[summary["plot_setting"] == "global", "final_accuracy"].iloc[0])
    local = summary[summary["plot_setting"] != "global"].sort_values("plot_k")

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    ax.plot(
        local["plot_k"],
        local["final_accuracy"],
        color="#1f77b4",
        linewidth=2.3,
        marker="o",
        markersize=5.0,
        label="Local sigma",
    )
    ax.axhline(
        global_final,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label=f"Global ({global_final:.2f})",
    )

    ax.set_xticks(local["plot_k"].tolist())
    ax.set_xticklabels([str(k) for k in local["plot_k"]], rotation=45, ha="right")
    ax.set_ylim(min(local["final_accuracy"].min(), global_final) - 0.7, max(local["final_accuracy"].max(), global_final) + 0.7)
    ax.set_xlabel("Local sigma K")
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_title("Tiny ImageNet MaxHerding + 1NN: final accuracy by exact K")
    ax.grid(True, alpha=0.25)
    ax.legend()

    out = PLOTS_DIR / "tinyimagenet_maor_final_accuracy_vs_k.png"
    _finish(fig, out)
    return out


def plot_effective_degree_vs_accuracy(summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    for _, row in summary.iterrows():
        setting = row["plot_setting"]
        color = "#d62728" if setting == "global" else "#1f77b4"
        ax.scatter(row["effective_degree_mean"], row["final_accuracy"], s=58, color=color)
        ax.annotate(
            "global" if setting == "global" else setting,
            (row["effective_degree_mean"], row["final_accuracy"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xlabel("Dataset mean effective degree")
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_title("Tiny ImageNet: effective graph degree vs final accuracy")
    ax.grid(True, alpha=0.25)

    out = PLOTS_DIR / "tinyimagenet_maor_effective_degree_vs_final_accuracy.png"
    _finish(fig, out)
    return out


def plot_selected_density_ratio_avg(summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10.6, 5.7))
    x = range(len(summary))
    labels = summary["plot_setting"].tolist()
    colors = ["#d62728" if label == "global" else "#4c78a8" for label in labels]
    ax.bar(x, summary["selected_to_dataset_effective_degree_ratio_avg"], color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Dataset average")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Selected / dataset effective degree")
    ax.set_title("Tiny ImageNet: average density ratio of selected batches")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    out = PLOTS_DIR / "tinyimagenet_maor_selected_density_ratio_avg.png"
    _finish(fig, out)
    return out


def plot_selected_density_ratio_by_episode(ratio: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    settings = sorted(ratio["setting_name"].unique(), key=_setting_sort_key)
    local_settings = [s for s in settings if s != "global"]
    cmap = plt.get_cmap("viridis")

    global_df = ratio[ratio["setting_name"] == "global"].sort_values("labels_selected_so_far")
    ax.plot(
        global_df["labels_selected_so_far"],
        global_df["selected_to_dataset_effective_degree_ratio"],
        color="#d62728",
        linewidth=2.8,
        marker="o",
        markersize=4.5,
        label="Global",
    )

    for idx, setting in enumerate(local_settings):
        group = ratio[ratio["setting_name"] == setting].sort_values("labels_selected_so_far")
        color = cmap(idx / max(1, len(local_settings) - 1))
        ax.plot(
            group["labels_selected_so_far"],
            group["selected_to_dataset_effective_degree_ratio"],
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3.8,
            label=setting,
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.1, label="Dataset average")
    ax.set_xticks(sorted(ratio["labels_selected_so_far"].unique()))
    ax.set_xlabel("Number of labeled points selected so far")
    ax.set_ylabel("Selected / dataset effective degree")
    ax.set_title("Tiny ImageNet: selected-batch density ratio by budget")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)

    out = PLOTS_DIR / "tinyimagenet_maor_selected_density_ratio_by_budget.png"
    _finish(fig, out)
    return out


def plot_selected_density_ratio_final(ratio: pd.DataFrame) -> Path:
    last = ratio.sort_values("labels_selected_so_far").groupby("setting_name", as_index=False).tail(1)
    last = last.sort_values("setting_name", key=lambda s: s.map(_setting_sort_key))

    fig, ax = plt.subplots(figsize=(10.6, 5.7))
    x = range(len(last))
    labels = last["setting_name"].tolist()
    colors = ["#d62728" if label == "global" else "#4c78a8" for label in labels]
    ax.bar(x, last["selected_to_dataset_effective_degree_ratio"], color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Dataset average")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Selected / dataset effective degree")
    ax.set_title("Tiny ImageNet: density ratio in last selected batch")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    out = PLOTS_DIR / "tinyimagenet_maor_selected_density_ratio_final_batch.png"
    _finish(fig, out)
    return out


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    acc = _load_accuracy()
    summary = _load_summary()
    ratio = pd.read_csv(RESULTS_DIR / "selected_density_ratio_by_episode.csv")

    paths = [
        plot_all_k_curves(acc),
        plot_best_k_vs_global(acc, summary),
        plot_final_accuracy_vs_k(summary),
        plot_effective_degree_vs_accuracy(summary),
        plot_selected_density_ratio_avg(summary),
        plot_selected_density_ratio_by_episode(ratio),
        plot_selected_density_ratio_final(ratio),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
