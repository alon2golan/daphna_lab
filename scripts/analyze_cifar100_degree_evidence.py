#!/usr/bin/env python3
"""CIFAR100-focused evidence for degree-driven MaxHerding behavior.

This script addresses the remaining research questions without launching new
active-learning runs. It aggregates the completed CIFAR100 MaxHerding K sweep,
then creates diagnostics that test whether performance is organized better by
the induced graph degree than by K itself.

Outputs are written under:

* ``results_summaries/cifar100_degree_evidence_2026_06_27``
* ``visual_outputs/cifar100_degree_evidence_2026_06_27``
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "deep_al") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "deep_al"))

from pycls.datasets.utils.features import load_features  # noqa: E402


DEFAULT_OUTDIR = Path("results_summaries/cifar100_degree_evidence_2026_06_27")
DEFAULT_PLOTS_DIR = Path("visual_outputs/cifar100_degree_evidence_2026_06_27")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--plots-dir", default=str(DEFAULT_PLOTS_DIR))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--degree-grid-sample-size", type=int, default=1000)
    parser.add_argument("--degree-grid-batch-size", type=int, default=64)
    parser.add_argument("--skip-degree-grid", action="store_true")
    parser.add_argument("--skip-clusters", action="store_true")
    return parser.parse_args()


def setting_name(local_sigma: bool, local_k: Any) -> str:
    if not bool(local_sigma):
        return "global"
    return f"K={int(float(local_k))}"


def setting_sort_key(setting: str) -> tuple[int, int]:
    if setting == "global":
        return (0, 0)
    return (1, int(setting.split("=")[1]))


def add_setting(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["setting"] = df.apply(
        lambda row: setting_name(row["local_sigma"], row.get("local_k")), axis=1
    )
    return df


def load_cifar100_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selection = pd.read_csv("results_summaries/maxherding_density_selection/selection_episode_summary.csv")
    density = pd.read_csv("results_summaries/maxherding_density_selection/density_summary.csv")
    remaining = pd.read_csv(
        "results_summaries/invariant_candidates_2026_06_27/remaining_degree_at_best_setting_mean.csv"
    )

    selection = add_setting(selection[selection["dataset"] == "CIFAR100"])
    density = add_setting(density[density["dataset"] == "CIFAR100"])
    remaining = remaining[remaining["dataset"] == "CIFAR100"].copy()
    selection["budget_labels"] = (selection["episode"].astype(int) + 1) * selection[
        "budget_size"
    ].astype(int)
    return selection, density, remaining


def summarize_accuracy_by_degree(selection: pd.DataFrame, density: pd.DataFrame) -> pd.DataFrame:
    accuracy = (
        selection.groupby(["budget_labels", "setting"], as_index=False)
        .agg(
            mean_accuracy=("test_accuracy_after_episode", "mean"),
            std_accuracy=("test_accuracy_after_episode", "std"),
            seeds=("seed", "nunique"),
        )
    )
    density_summary = (
        density.groupby("setting", as_index=False)
        .agg(
            dataset_effective_degree=("effective_degree_mean", "mean"),
            dataset_raw_degree=("raw_degree_mean", "mean"),
            sigma_mean=("sigma_mean", "mean"),
            sigma_cv=("sigma_cv", "mean"),
        )
    )
    merged = accuracy.merge(density_summary, on="setting", how="left")
    merged["local_k"] = merged["setting"].map(
        lambda value: math.nan if value == "global" else int(value.split("=")[1])
    )
    merged["log_dataset_effective_degree"] = np.log10(
        merged["dataset_effective_degree"].clip(lower=1e-8)
    )
    return merged.sort_values(["budget_labels", "dataset_effective_degree"])


def rank_settings_by_budget(accuracy_degree: pd.DataFrame) -> pd.DataFrame:
    ranked = accuracy_degree.sort_values(
        ["budget_labels", "mean_accuracy"], ascending=[True, False]
    ).copy()
    ranked["rank"] = ranked.groupby("budget_labels").cumcount() + 1
    return ranked


def budget_correlations(accuracy_degree: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for budget, group in accuracy_degree.groupby("budget_labels"):
        rows.append(
            {
                "budget_labels": int(budget),
                "pearson_accuracy_vs_degree": group["mean_accuracy"].corr(
                    group["dataset_effective_degree"], method="pearson"
                ),
                "spearman_accuracy_vs_degree": group["mean_accuracy"].corr(
                    group["dataset_effective_degree"], method="spearman"
                ),
                "pearson_accuracy_vs_log_degree": group["mean_accuracy"].corr(
                    group["log_dataset_effective_degree"], method="pearson"
                ),
                "spearman_accuracy_vs_log_degree": group["mean_accuracy"].corr(
                    group["log_dataset_effective_degree"], method="spearman"
                ),
            }
        )
    return pd.DataFrame(rows)


def selected_degree_ratio_at_best(
    selection: pd.DataFrame, density: pd.DataFrame, ranked: pd.DataFrame
) -> pd.DataFrame:
    best = ranked[ranked["rank"] == 1][["budget_labels", "setting", "mean_accuracy"]]
    density_summary = density.groupby("setting", as_index=False).agg(
        dataset_effective_degree=("effective_degree_mean", "mean"),
        dataset_raw_degree=("raw_degree_mean", "mean"),
    )
    selected = (
        selection.groupby(["budget_labels", "setting"], as_index=False)
        .agg(
            selected_effective_degree=("selected_effective_degree_mean", "mean"),
            selected_raw_degree=("selected_raw_degree_mean", "mean"),
            selected_classes=("unique_selected_classes", "mean"),
            selected_entropy=("selected_label_entropy_normalized", "mean"),
        )
        .merge(best, on=["budget_labels", "setting"], how="inner")
        .merge(density_summary, on="setting", how="left")
    )
    selected["selected_to_dataset_effective_degree_ratio"] = (
        selected["selected_effective_degree"] / selected["dataset_effective_degree"]
    )
    selected["selected_to_dataset_raw_degree_ratio"] = (
        selected["selected_raw_degree"] / selected["dataset_raw_degree"]
    )
    return selected.sort_values("budget_labels")


def selected_degree_ratio_for_fixed_settings(
    selection: pd.DataFrame, density: pd.DataFrame, settings: list[str]
) -> pd.DataFrame:
    density_summary = density.groupby("setting", as_index=False).agg(
        dataset_effective_degree=("effective_degree_mean", "mean"),
        dataset_raw_degree=("raw_degree_mean", "mean"),
    )
    selected = (
        selection.groupby(["budget_labels", "setting"], as_index=False)
        .agg(
            selected_effective_degree=("selected_effective_degree_mean", "mean"),
            selected_effective_degree_std=("selected_effective_degree_mean", "std"),
            selected_raw_degree=("selected_raw_degree_mean", "mean"),
            mean_accuracy=("test_accuracy_after_episode", "mean"),
            seeds=("seed", "nunique"),
        )
        .merge(density_summary, on="setting", how="left")
    )
    selected["selected_to_dataset_effective_degree_ratio"] = (
        selected["selected_effective_degree"] / selected["dataset_effective_degree"]
    )
    return selected[selected["setting"].isin(settings)].sort_values(["setting", "budget_labels"])


def stability_summary(ranked: pd.DataFrame, remaining: pd.DataFrame) -> pd.DataFrame:
    best = ranked[ranked["rank"] == 1].sort_values("budget_labels").copy()
    dataset_degree = best["dataset_effective_degree"]
    rem_degree = remaining["remaining_effective_degree_mean_mean"]
    rows = [
        {
            "metric": "best_setting_dataset_effective_degree",
            "mean": dataset_degree.mean(),
            "std": dataset_degree.std(),
            "cv": dataset_degree.std() / dataset_degree.mean(),
            "min": dataset_degree.min(),
            "max": dataset_degree.max(),
            "range_pct_of_mean": (dataset_degree.max() - dataset_degree.min())
            / dataset_degree.mean(),
        },
        {
            "metric": "best_setting_remaining_effective_degree",
            "mean": rem_degree.mean(),
            "std": rem_degree.std(),
            "cv": rem_degree.std() / rem_degree.mean(),
            "min": rem_degree.min(),
            "max": rem_degree.max(),
            "range_pct_of_mean": (rem_degree.max() - rem_degree.min()) / rem_degree.mean(),
        },
    ]
    return pd.DataFrame(rows)


def load_cached_sigmas(k: int) -> np.ndarray:
    candidates = [
        Path("results_summaries/maxherding_density_selection/cache")
        / f"local_sigmas_CIFAR100_norm_default_k{k}.npy",
        Path("results_summaries/invariant_candidates_2026_06_27/cache")
        / f"local_sigmas_CIFAR100_norm_default_k{k}.npy",
    ]
    for path in candidates:
        if path.exists():
            return np.load(path)
    raise FileNotFoundError(f"No cached local sigma file found for K={k}")


def estimate_degree_grid(
    *,
    sample_size: int,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    deltas = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    k_values = [480, 960, 1440, 2000, 3500, 5000]
    rng = np.random.default_rng(0)
    features = load_features("CIFAR100", train=True, normalized=True).astype(np.float32, copy=False)
    anchors = np.sort(rng.choice(len(features), size=min(sample_size, len(features)), replace=False))
    feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    anchor_tensor = torch.as_tensor(anchors, dtype=torch.long, device=device)

    rows = []
    sigma_by_setting: dict[str, np.ndarray | None] = {"global": None}
    for k in k_values:
        sigma_by_setting[f"K={k}"] = load_cached_sigmas(k).astype(np.float32, copy=False)

    with torch.no_grad():
        for setting, sigmas in sigma_by_setting.items():
            counts_by_delta = {delta: [] for delta in deltas}
            sigma_all = None
            if sigmas is not None:
                sigma_all = torch.as_tensor(sigmas, dtype=torch.float32, device=device)

            for start in range(0, len(anchors), batch_size):
                batch_idx = anchor_tensor[start : start + batch_size]
                batch_anchor_np = anchors[start : start + batch_size]
                dist = torch.cdist(feature_tensor[batch_idx], feature_tensor)
                if sigma_all is None:
                    scaled = dist
                    for delta in deltas:
                        count = torch.count_nonzero(scaled < delta, dim=1) - 1
                        counts_by_delta[delta].append(count.detach().cpu().numpy())
                else:
                    sigma_i = sigma_all[batch_idx].unsqueeze(1)
                    denom = torch.clamp(sigma_i * sigma_all.unsqueeze(0), min=1e-12)
                    scaled_sq = (dist * dist) / denom
                    for delta in deltas:
                        count = torch.count_nonzero(scaled_sq < delta * delta, dim=1) - 1
                        counts_by_delta[delta].append(count.detach().cpu().numpy())
                del dist

            for delta in deltas:
                values = np.concatenate(counts_by_delta[delta]).astype(np.float64)
                rows.append(
                    {
                        "setting": setting,
                        "local_k": "" if setting == "global" else int(setting.split("=")[1]),
                        "delta": delta,
                        "sampled_anchors": len(anchors),
                        "effective_degree_mean": float(values.mean()),
                        "effective_degree_std": float(values.std()),
                        "effective_degree_cv": float(values.std() / values.mean())
                        if values.mean()
                        else math.nan,
                        "effective_degree_median": float(np.median(values)),
                    }
                )
    return pd.DataFrame(rows)


def cluster_separability_by_k() -> pd.DataFrame:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import pairwise_distances, silhouette_score
    from torchvision import datasets

    features = load_features("CIFAR100", train=True, normalized=True).astype(np.float32, copy=False)
    labels = np.asarray(
        datasets.CIFAR100(root="data", train=True, download=False).targets, dtype=np.int64
    )
    rows = []
    for k in [10, 20, 50, 100, 200]:
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=0,
            batch_size=4096,
            n_init=5,
            max_iter=200,
        )
        cluster_ids = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_.astype(np.float32, copy=False)
        distances_to_center = np.linalg.norm(features - centers[cluster_ids], axis=1)
        centroid_dist = pairwise_distances(centers)
        inter = centroid_dist[np.triu_indices(k, 1)]
        sizes = np.bincount(cluster_ids, minlength=k).astype(np.float64)
        majority_counts = []
        for cluster_id in range(k):
            cluster_labels = labels[cluster_ids == cluster_id]
            if cluster_labels.size == 0:
                majority_counts.append(0)
                continue
            majority_counts.append(np.bincount(cluster_labels, minlength=100).max())
        purity = float(np.sum(majority_counts) / len(labels))
        sil = silhouette_score(features, cluster_ids, sample_size=5000, random_state=0)
        rows.append(
            {
                "num_clusters": k,
                "inertia_per_point": float(kmeans.inertia_ / len(features)),
                "intra_cluster_distance_mean": float(distances_to_center.mean()),
                "inter_centroid_distance_mean": float(inter.mean()),
                "cluster_separability_ratio": float(inter.mean() / distances_to_center.mean()),
                "silhouette_sample_5000": float(sil),
                "cluster_size_min": int(sizes.min()),
                "cluster_size_max": int(sizes.max()),
                "cluster_size_cv": float(sizes.std() / sizes.mean()),
                "true_label_cluster_purity": purity,
            }
        )
    return pd.DataFrame(rows)


def save_accuracy_degree_plots(
    accuracy_degree: pd.DataFrame,
    ranked: pd.DataFrame,
    remaining: pd.DataFrame,
    selected_best: pd.DataFrame,
    selected_fixed: pd.DataFrame,
    degree_grid: pd.DataFrame | None,
    cluster_sep: pd.DataFrame | None,
    plots_dir: Path,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    budgets_to_show = [100, 300, 600, 900]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for budget in budgets_to_show:
        group = accuracy_degree[accuracy_degree["budget_labels"] == budget].sort_values(
            "dataset_effective_degree"
        )
        ax.plot(
            group["dataset_effective_degree"],
            group["mean_accuracy"],
            marker="o",
            label=f"{budget} labels",
        )
    ax.axvspan(
        19,
        25,
        color="#2ca02c",
        alpha=0.12,
        label="post-hoc range of best settings",
    )
    ax.set_xlabel("Dataset effective degree")
    ax.set_ylabel("Accuracy")
    ax.set_title("CIFAR100 Accuracy vs Dataset Effective Degree")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Budget", loc="best")
    fig.tight_layout()
    fig.savefig(plots_dir / "cifar100_accuracy_vs_effective_degree_by_budget.png")
    plt.close(fig)

    top3 = ranked[ranked["rank"] <= 3].copy()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for rank, group in top3.groupby("rank"):
        ax.plot(
            group["budget_labels"],
            group["dataset_effective_degree"],
            marker="o",
            label=f"rank {rank}",
        )
    ax.set_xlabel("Labeled points")
    ax.set_ylabel("Dataset effective degree")
    ax.set_title("CIFAR100 Dataset Effective Degree of Top-Ranked Settings")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Accuracy rank", loc="upper right")
    fig.tight_layout()
    fig.savefig(plots_dir / "cifar100_top3_setting_degree_by_budget.png")
    plt.close(fig)

    best = ranked[ranked["rank"] == 1].sort_values("budget_labels")
    fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
    ax1.plot(
        best["budget_labels"],
        best["dataset_effective_degree"],
        marker="o",
        color="#1f77b4",
        label="Best-setting dataset degree",
    )
    ax1.plot(
        remaining["budget_labels"],
        remaining["remaining_effective_degree_mean_mean"],
        marker="s",
        color="#ff7f0e",
        label="Remaining-pool degree",
    )
    for row in best.itertuples(index=False):
        ax1.annotate(
            row.setting,
            (row.budget_labels, row.dataset_effective_degree),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )
    ax1.set_xlabel("Labeled points")
    ax1.set_ylabel("Effective degree")
    ax1.set_title("CIFAR100 Best-Setting and Remaining-Pool Effective Degree")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(plots_dir / "cifar100_best_setting_degree_and_remaining_degree.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(
        selected_best["budget_labels"],
        selected_best["selected_to_dataset_effective_degree_ratio"],
        marker="o",
        color="#d62728",
        label="Selected batch / dataset",
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Dataset average")
    ax.set_xlabel("Labeled points")
    ax.set_ylabel("Effective-degree ratio")
    ax.set_title("CIFAR100 Selected-to-Dataset Effective Degree Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plots_dir / "cifar100_selected_to_dataset_degree_ratio_best_setting.png")
    plt.close(fig)

    if not selected_fixed.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        for setting, group in selected_fixed.groupby("setting"):
            group = group.sort_values("budget_labels")
            ax.plot(
                group["budget_labels"],
                group["selected_to_dataset_effective_degree_ratio"],
                marker="o",
                label=setting,
            )
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Dataset average")
        ax.set_xlabel("Labeled points")
        ax.set_ylabel("Selected batch / dataset effective degree")
        ax.set_title("CIFAR100 Selected-to-Dataset Degree Ratio for Fixed Settings")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plots_dir / "cifar100_selected_to_dataset_degree_ratio_fixed_settings.png")
        plt.close(fig)

    if degree_grid is not None and not degree_grid.empty:
        pivot = degree_grid[degree_grid["setting"] != "global"].pivot(
            index="delta", columns="setting", values="effective_degree_mean"
        )
        ordered_cols = sorted(pivot.columns, key=lambda value: int(value.split("=")[1]))
        pivot = pivot[ordered_cols]
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        image = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{value:.2f}" for value in pivot.index])
        ax.set_xlabel("Local K")
        ax.set_ylabel("Delta")
        ax.set_title("CIFAR100 Estimated Dataset Effective Degree by Delta and K")
        fig.colorbar(image, ax=ax, label="Estimated dataset effective degree")
        fig.tight_layout()
        fig.savefig(plots_dir / "cifar100_delta_k_effective_degree_grid.png")
        plt.close(fig)

    if cluster_sep is not None and not cluster_sep.empty:
        fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
        ax1.plot(
            cluster_sep["num_clusters"],
            cluster_sep["cluster_separability_ratio"],
            marker="o",
            label="Separability ratio",
        )
        ax1.set_xlabel("Number of k-means clusters")
        ax1.set_ylabel("Inter-centroid / intra-cluster distance")
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(
            cluster_sep["num_clusters"],
            cluster_sep["true_label_cluster_purity"],
            marker="s",
            color="#d62728",
            label="True-label purity",
        )
        ax2.set_ylabel("True-label purity")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        ax1.set_title("CIFAR100 Feature Clustering Metrics by Number of Clusters")
        fig.tight_layout()
        fig.savefig(plots_dir / "cifar100_cluster_separability_by_k.png")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    plots_dir = Path(args.plots_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    selection, density, remaining = load_cifar100_tables()
    accuracy_degree = summarize_accuracy_by_degree(selection, density)
    ranked = rank_settings_by_budget(accuracy_degree)
    top_settings = ranked[ranked["rank"] <= 3].copy()
    correlations = budget_correlations(accuracy_degree)
    selected_best = selected_degree_ratio_at_best(selection, density, ranked)
    selected_fixed = selected_degree_ratio_for_fixed_settings(
        selection, density, settings=["global", "K=1440", "K=2000"]
    )
    stability = stability_summary(ranked, remaining)

    accuracy_degree.to_csv(outdir / "cifar100_accuracy_by_budget_and_degree.csv", index=False)
    top_settings.to_csv(outdir / "cifar100_top3_settings_by_budget.csv", index=False)
    correlations.to_csv(outdir / "cifar100_accuracy_degree_correlations_by_budget.csv", index=False)
    selected_best.to_csv(outdir / "cifar100_selected_degree_ratio_at_best_setting.csv", index=False)
    selected_fixed.to_csv(outdir / "cifar100_selected_degree_ratio_fixed_settings.csv", index=False)
    stability.to_csv(outdir / "cifar100_best_degree_stability_summary.csv", index=False)

    degree_grid = None
    if not args.skip_degree_grid:
        degree_grid = estimate_degree_grid(
            sample_size=args.degree_grid_sample_size,
            batch_size=args.degree_grid_batch_size,
            device=args.device,
        )
        degree_grid.to_csv(outdir / "cifar100_delta_k_effective_degree_grid.csv", index=False)
    elif (outdir / "cifar100_delta_k_effective_degree_grid.csv").exists():
        degree_grid = pd.read_csv(outdir / "cifar100_delta_k_effective_degree_grid.csv")

    cluster_sep = None
    if not args.skip_clusters:
        cluster_sep = cluster_separability_by_k()
        cluster_sep.to_csv(outdir / "cifar100_cluster_separability_by_k.csv", index=False)
    elif (outdir / "cifar100_cluster_separability_by_k.csv").exists():
        cluster_sep = pd.read_csv(outdir / "cifar100_cluster_separability_by_k.csv")

    save_accuracy_degree_plots(
        accuracy_degree=accuracy_degree,
        ranked=ranked,
        remaining=remaining,
        selected_best=selected_best,
        selected_fixed=selected_fixed,
        degree_grid=degree_grid,
        cluster_sep=cluster_sep,
        plots_dir=plots_dir,
    )

    print(f"Wrote CSVs under {outdir}")
    print(f"Wrote plots under {plots_dir}")


if __name__ == "__main__":
    main()
