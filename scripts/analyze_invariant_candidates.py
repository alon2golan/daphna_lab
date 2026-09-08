#!/usr/bin/env python3
"""Analyze budget-robust invariant candidates from completed MaxHerding runs.

This is an offline diagnostic script. It does not run active learning or train a
classifier. It reads completed MaxHerding output folders and asks:

* Which local-sigma setting was best at each budget?
* At that best setting, what is the average graph degree of the remaining
  unlabeled pool?
* Do simple dataset-level quantities, such as number of classes and class
  separability in feature space, line up with the useful degree ranges?

The remaining-pool degree is estimated by sampling anchors from each saved
``uSet.npy`` and counting their neighbors inside the same remaining pool. This
keeps the calculation practical while still measuring the quantity requested:
the graph induced by the nodes that are still unlabeled at each budget.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "deep_al") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "deep_al"))

from pycls.al.kernel_utils import compute_local_sigmas  # noqa: E402
from pycls.datasets.utils.features import load_features  # noqa: E402


DENSITY_SUMMARIES = [
    Path("results_summaries/maxherding_density_selection/density_summary.csv"),
    Path("results_summaries/maxherding_tinyimagenet_maor_density_selection/density_summary.csv"),
]
SELECTION_SUMMARIES = [
    Path("results_summaries/maxherding_density_selection/selection_episode_summary.csv"),
    Path("results_summaries/maxherding_tinyimagenet_maor_density_selection/selection_episode_summary.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute invariant-candidate diagnostics from saved MaxHerding runs."
    )
    parser.add_argument(
        "--outdir",
        default="results_summaries/invariant_candidates_2026_06_27",
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--plots-dir",
        default="visual_outputs/invariant_candidates_2026_06_27",
        help="Directory for plots.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=512,
        help="Number of remaining-pool anchors sampled per run and episode.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed for deterministic anchor sampling.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for distance computations.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Anchor batch size for torch.cdist.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["CIFAR10", "CIFAR100", "TINYIMAGENET"],
        help="Datasets to include.",
    )
    return parser.parse_args()


def get_config_value(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def read_config(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def setting_name(local_sigma: bool, local_k: float | int | str | None) -> str:
    if not local_sigma:
        return "global"
    return f"K={int(float(local_k))}"


def setting_sort_key(setting: str) -> tuple[int, int]:
    if setting == "global":
        return (0, 0)
    return (1, int(setting.split("=")[1]))


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_cv": math.nan,
            f"{prefix}_q25": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_q75": math.nan,
        }
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_cv": float(std / mean) if mean else math.nan,
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
    }


def load_combined_selection() -> pd.DataFrame:
    frames = []
    for path in SELECTION_SUMMARIES:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No selection summary CSVs were found.")
    df = pd.concat(frames, ignore_index=True)
    df["setting"] = df.apply(
        lambda row: setting_name(bool(row["local_sigma"]), row.get("local_k")), axis=1
    )
    # The older diagnostic CSV stored plot x-values as episode indices in some
    # runs. The saved budget size is reliable, so derive the label budget here.
    df["budget_labels"] = (df["episode"].astype(int) + 1) * df["budget_size"].astype(int)
    return df


def load_combined_density() -> pd.DataFrame:
    frames = []
    for path in DENSITY_SUMMARIES:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No density summary CSVs were found.")
    df = pd.concat(frames, ignore_index=True)
    df["setting"] = df.apply(
        lambda row: setting_name(bool(row["local_sigma"]), row.get("local_k")), axis=1
    )
    return df


def choose_best_settings(selection: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        selection.groupby(["dataset", "budget_labels", "setting"], as_index=False)
        .agg(
            mean_accuracy=("test_accuracy_after_episode", "mean"),
            std_accuracy=("test_accuracy_after_episode", "std"),
            seeds=("seed", "nunique"),
        )
        .sort_values(["dataset", "budget_labels", "mean_accuracy"], ascending=[True, True, False])
    )
    best = grouped.groupby(["dataset", "budget_labels"], as_index=False).head(1).copy()
    best = best.sort_values(["dataset", "budget_labels"]).reset_index(drop=True)
    return best


def load_labels(dataset: str, data_root: str) -> np.ndarray:
    from torchvision import datasets

    if dataset == "CIFAR10":
        ds = datasets.CIFAR10(root=data_root, train=True, download=False)
    elif dataset == "CIFAR100":
        ds = datasets.CIFAR100(root=data_root, train=True, download=False)
    elif dataset == "TINYIMAGENET":
        from pycls.datasets.tiny_imagenet import TinyImageNet

        ds = TinyImageNet(root=data_root, split="train")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return np.asarray(ds.targets, dtype=np.int64)


def get_local_sigmas(
    *,
    features: np.ndarray,
    dataset: str,
    normalized: bool,
    features_tag: str | None,
    local_k: int,
    cache_dir: Path,
    device: str,
) -> np.ndarray:
    tag = features_tag or "default"
    norm = "norm" if normalized else "unnorm"
    cache_path = cache_dir / f"local_sigmas_{dataset}_{norm}_{tag}_k{local_k}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    legacy_cache_paths = [
        Path("results_summaries/maxherding_density_selection/cache")
        / f"local_sigmas_{dataset}_{norm}_{tag}_k{local_k}.npy",
        Path("results_summaries/maxherding_tinyimagenet_maor_density_selection/cache")
        / f"local_sigmas_{dataset}_{norm}_{tag}_k{local_k}.npy",
    ]
    for legacy_path in legacy_cache_paths:
        if legacy_path.exists():
            sigmas = np.load(legacy_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, sigmas)
            return sigmas
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sigmas = compute_local_sigmas(
        features,
        k=local_k,
        batch_size=512,
        device=device,
        dtype=torch.float32,
    ).numpy()
    np.save(cache_path, sigmas)
    return sigmas


def count_remaining_neighbors(
    *,
    features: np.ndarray,
    anchors: np.ndarray,
    targets: np.ndarray,
    delta: float,
    global_sigma: float,
    local_sigmas: np.ndarray | None,
    device: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    target_idx = torch.as_tensor(targets, dtype=torch.long, device=device)
    target_features = feature_tensor[target_idx]
    target_set = set(int(x) for x in targets.tolist())
    sigma_targets = None
    if local_sigmas is not None:
        sigma_targets = torch.as_tensor(local_sigmas[targets], dtype=torch.float32, device=device)

    raw_counts = []
    effective_counts = []
    delta_sq = float(delta) ** 2
    with torch.no_grad():
        for start in range(0, len(anchors), batch_size):
            batch_np = anchors[start : start + batch_size]
            batch_idx = torch.as_tensor(batch_np, dtype=torch.long, device=device)
            dist = torch.cdist(feature_tensor[batch_idx], target_features)
            raw = torch.count_nonzero(dist < float(delta), dim=1)
            if local_sigmas is None:
                effective = torch.count_nonzero((dist / float(global_sigma)) < float(delta), dim=1)
            else:
                sigma_i = torch.as_tensor(
                    local_sigmas[batch_np], dtype=torch.float32, device=device
                ).unsqueeze(1)
                denom = torch.clamp(sigma_i * sigma_targets.unsqueeze(0), min=1e-12)
                effective = torch.count_nonzero((dist * dist) / denom < delta_sq, dim=1)

            # Anchors are sampled from targets, so remove the self-edge.
            self_mask = np.asarray([int(idx) in target_set for idx in batch_np], dtype=bool)
            if self_mask.any():
                raw = raw - torch.as_tensor(self_mask, dtype=raw.dtype, device=device)
                effective = effective - torch.as_tensor(
                    self_mask, dtype=effective.dtype, device=device
                )
            raw_counts.append(raw.detach().cpu().numpy())
            effective_counts.append(effective.detach().cpu().numpy())
            del dist

    return {
        "remaining_raw_degree": np.concatenate(raw_counts).astype(np.float64),
        "remaining_effective_degree": np.concatenate(effective_counts).astype(np.float64),
    }


def parse_run_meta(run_dir: Path) -> dict[str, Any]:
    config = read_config(run_dir)
    dataset = get_config_value(config, "DATASET.NAME")
    seed = int(get_config_value(config, "RNG_SEED", -1))
    local_sigma = bool(
        get_config_value(config, "LOCAL_SIGMA", get_config_value(config, "ACTIVE_LEARNING.LOCAL_SIGMA", False))
    )
    local_k = get_config_value(
        config, "LOCAL_SIGMA_K", get_config_value(config, "ACTIVE_LEARNING.LOCAL_SIGMA_K", None)
    )
    return {
        "config": config,
        "dataset": dataset,
        "seed": seed,
        "local_sigma": local_sigma,
        "local_k": int(local_k) if local_sigma and local_k is not None else None,
        "setting": setting_name(local_sigma, local_k),
        "delta": float(get_config_value(config, "ACTIVE_LEARNING.INITIAL_DELTA")),
        "global_sigma": float(get_config_value(config, "ACTIVE_LEARNING.INITIAL_SIGMA", 1.0)),
        "features_normalized": bool(
            get_config_value(
                config, "FEATURES_NORMALIZED", get_config_value(config, "DATASET.FEATURES_NORMALIZED", True)
            )
        ),
        "features_tag": get_config_value(
            config, "FEATURES_TAG", get_config_value(config, "DATASET.FEATURES_TAG", None)
        ),
        "data_root": get_config_value(config, "DATASET.ROOT_DIR"),
    }


def episode_dirs(run_dir: Path) -> list[Path]:
    pattern = re.compile(r"episode_(\d+)$")
    dirs = [path for path in run_dir.iterdir() if path.is_dir() and pattern.match(path.name)]
    return sorted(dirs, key=lambda path: int(pattern.match(path.name).group(1)))


def load_index_array(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=True)
    return np.asarray(values, dtype=np.int64).reshape(-1)


def compute_remaining_degree(
    *,
    selection: pd.DataFrame,
    best: pd.DataFrame,
    outdir: Path,
    sample_size: int,
    sample_seed: int,
    device: str,
    batch_size: int,
    datasets: set[str],
) -> pd.DataFrame:
    needed = set(
        (row.dataset, int(row.budget_labels), row.setting)
        for row in best.itertuples(index=False)
        if row.dataset in datasets
    )
    candidate_runs = (
        selection[["dataset", "seed", "setting", "run_dir"]]
        .drop_duplicates()
        .query("dataset in @datasets")
    )

    feature_cache: dict[tuple[str, bool, str | None], np.ndarray] = {}
    sigma_cache: dict[tuple[str, bool, str | None, int], np.ndarray] = {}
    label_cache: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, Any]] = []

    for run in candidate_runs.itertuples(index=False):
        run_dir = Path(run.run_dir)
        if not run_dir.exists():
            continue
        meta = parse_run_meta(run_dir)
        if not any(
            (meta["dataset"], budget, meta["setting"]) in needed
            for budget in best.loc[best["dataset"] == meta["dataset"], "budget_labels"].astype(int)
        ):
            continue

        feature_key = (
            meta["dataset"],
            meta["features_normalized"],
            meta["features_tag"],
        )
        if feature_key not in feature_cache:
            feature_cache[feature_key] = load_features(
                meta["dataset"],
                train=True,
                normalized=meta["features_normalized"],
                features_tag=meta["features_tag"],
            ).astype(np.float32, copy=False)
        features = feature_cache[feature_key]

        sigmas = None
        if meta["local_sigma"]:
            sigma_key = (*feature_key, int(meta["local_k"]))
            if sigma_key not in sigma_cache:
                sigma_cache[sigma_key] = get_local_sigmas(
                    features=features,
                    dataset=meta["dataset"],
                    normalized=meta["features_normalized"],
                    features_tag=meta["features_tag"],
                    local_k=int(meta["local_k"]),
                    cache_dir=outdir / "cache",
                    device=device,
                )
            sigmas = sigma_cache[sigma_key]

        labels = None
        label_key = (meta["dataset"], meta["data_root"])
        if label_key not in label_cache:
            label_cache[label_key] = load_labels(meta["dataset"], meta["data_root"])
        labels = label_cache[label_key]
        num_classes = int(labels.max()) + 1

        for ep_dir in episode_dirs(run_dir):
            lset_path = ep_dir / "lSet.npy"
            uset_path = ep_dir / "uSet.npy"
            active_path = ep_dir / "activeSet.npy"
            if not lset_path.exists() or not uset_path.exists():
                continue
            lset = load_index_array(lset_path)
            uset = load_index_array(uset_path)
            budget = int(len(lset))
            key = (meta["dataset"], budget, meta["setting"])
            if key not in needed:
                continue

            seed_key = f"{sample_seed}:{meta['dataset']}:{meta['seed']}:{meta['setting']}:{budget}"
            rng_seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(rng_seed)
            anchor_count = min(sample_size, len(uset))
            anchors = np.sort(rng.choice(uset, size=anchor_count, replace=False))
            degrees = count_remaining_neighbors(
                features=features,
                anchors=anchors,
                targets=uset,
                delta=meta["delta"],
                global_sigma=meta["global_sigma"],
                local_sigmas=sigmas,
                device=device,
                batch_size=batch_size,
            )

            active = load_index_array(active_path) if active_path.exists() else np.asarray([])
            selected_classes = len(np.unique(labels[active])) if active.size else 0
            row = {
                "dataset": meta["dataset"],
                "seed": meta["seed"],
                "setting": meta["setting"],
                "budget_labels": budget,
                "remaining_nodes": int(len(uset)),
                "sampled_remaining_nodes": int(anchor_count),
                "delta": meta["delta"],
                "local_k": meta["local_k"] if meta["local_k"] is not None else "",
                "num_classes": num_classes,
                "selected_classes_this_episode": int(selected_classes),
            }
            row.update(summarize(degrees["remaining_raw_degree"], "remaining_raw_degree"))
            row.update(summarize(degrees["remaining_effective_degree"], "remaining_effective_degree"))
            row["remaining_effective_degree_fraction_mean"] = (
                row["remaining_effective_degree_mean"] / max(len(uset) - 1, 1)
            )
            rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    best_meta = best.rename(
        columns={
            "setting": "best_setting",
            "mean_accuracy": "best_mean_accuracy",
            "std_accuracy": "best_std_accuracy",
            "seeds": "best_accuracy_seeds",
        }
    )
    result = result.merge(
        best_meta,
        left_on=["dataset", "budget_labels", "setting"],
        right_on=["dataset", "budget_labels", "best_setting"],
        how="left",
    )
    return result.sort_values(["dataset", "budget_labels", "seed", "setting"])


def class_separability(
    *,
    selection: pd.DataFrame,
    datasets: set[str],
    device: str,
) -> pd.DataFrame:
    rows = []
    seen = set()
    for run in selection[["dataset", "run_dir"]].drop_duplicates().itertuples(index=False):
        if run.dataset not in datasets or run.dataset in seen:
            continue
        run_dir = Path(run.run_dir)
        if not run_dir.exists():
            continue
        meta = parse_run_meta(run_dir)
        features = load_features(
            meta["dataset"],
            train=True,
            normalized=meta["features_normalized"],
            features_tag=meta["features_tag"],
        ).astype(np.float32, copy=False)
        labels = load_labels(meta["dataset"], meta["data_root"])
        classes = np.unique(labels)
        centroids = np.stack([features[labels == cls].mean(axis=0) for cls in classes])
        intra = []
        for cls, centroid in zip(classes, centroids):
            cls_features = features[labels == cls]
            intra.append(np.linalg.norm(cls_features - centroid, axis=1).mean())
        intra_mean = float(np.mean(intra))

        centroid_tensor = torch.as_tensor(centroids, dtype=torch.float32, device=device)
        with torch.no_grad():
            centroid_dist = torch.cdist(centroid_tensor, centroid_tensor).cpu().numpy()
            feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            pred_chunks = []
            for start in range(0, len(features), 4096):
                dist = torch.cdist(feature_tensor[start : start + 4096], centroid_tensor)
                pred_chunks.append(torch.argmin(dist, dim=1).detach().cpu().numpy())
        upper = centroid_dist[np.triu_indices(len(classes), k=1)]
        pred = np.concatenate(pred_chunks)
        nearest_centroid_acc = float(np.mean(classes[pred] == labels))

        rows.append(
            {
                "dataset": meta["dataset"],
                "num_points": int(len(features)),
                "num_classes": int(len(classes)),
                "features_normalized": bool(meta["features_normalized"]),
                "features_tag": meta["features_tag"] or "",
                "class_intra_distance_mean": intra_mean,
                "class_inter_centroid_distance_mean": float(np.mean(upper)),
                "class_separability_ratio": float(np.mean(upper) / intra_mean)
                if intra_mean
                else math.nan,
                "nearest_centroid_train_accuracy": nearest_centroid_acc,
            }
        )
        seen.add(run.dataset)
    return pd.DataFrame(rows)


def plot_remaining_degree(summary: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return
    for dataset, df in summary.groupby("dataset"):
        fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
        ax1.plot(
            df["budget_labels"],
            df["remaining_effective_degree_mean_mean"],
            marker="o",
            color="#1f77b4",
            label="Remaining effective degree",
        )
        ax1.set_xlabel("Labeled points")
        ax1.set_ylabel("Average remaining-node effective degree")
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(
            df["budget_labels"],
            df["best_mean_accuracy_mean"],
            marker="s",
            linestyle="--",
            color="#d62728",
            label="Accuracy at best setting",
        )
        ax2.set_ylabel("Accuracy")
        for row in df.itertuples(index=False):
            ax1.annotate(
                row.setting,
                (row.budget_labels, row.remaining_effective_degree_mean_mean),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=2,
            frameon=False,
        )
        ax1.set_title(f"{dataset}: Remaining-Pool Degree at Per-Budget Best Local Setting")
        fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
        fig.savefig(plots_dir / f"{dataset.lower()}_best_setting_remaining_degree.png")
        plt.close(fig)


def plot_dataset_cross_summary(cross: pd.DataFrame, plots_dir: Path) -> None:
    if cross.empty:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.scatter(
        cross["class_separability_ratio"],
        cross["best_final_effective_degree"],
        s=90,
        color="#2ca02c",
    )
    for row in cross.itertuples(index=False):
        ax.annotate(
            f"{row.dataset}\n{int(row.num_classes)} classes",
            (row.class_separability_ratio, row.best_final_effective_degree),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )
    ax.set_xlabel("Class separability ratio: mean inter-centroid distance / mean intra-class distance")
    ax.set_ylabel("Effective graph degree of best final setting")
    ax.set_title("Dataset-level candidate invariant: separability vs useful graph degree")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "dataset_separability_vs_best_degree.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    plots_dir = Path(args.plots_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    datasets = set(args.datasets)

    selection = load_combined_selection()
    selection = selection[selection["dataset"].isin(datasets)].copy()
    density = load_combined_density()
    density = density[density["dataset"].isin(datasets)].copy()

    best = choose_best_settings(selection)
    best.to_csv(outdir / "best_setting_by_budget.csv", index=False)

    remaining = compute_remaining_degree(
        selection=selection,
        best=best,
        outdir=outdir,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        device=args.device,
        batch_size=args.batch_size,
        datasets=datasets,
    )
    remaining.to_csv(outdir / "remaining_degree_at_best_setting_by_seed.csv", index=False)

    if not remaining.empty:
        remaining_summary = (
            remaining.groupby(["dataset", "budget_labels", "setting"], as_index=False)
            .agg(
                remaining_effective_degree_mean_mean=("remaining_effective_degree_mean", "mean"),
                remaining_effective_degree_mean_std=("remaining_effective_degree_mean", "std"),
                remaining_effective_degree_cv_mean=("remaining_effective_degree_cv", "mean"),
                remaining_effective_degree_fraction_mean=(
                    "remaining_effective_degree_fraction_mean",
                    "mean",
                ),
                remaining_raw_degree_mean_mean=("remaining_raw_degree_mean", "mean"),
                best_mean_accuracy_mean=("best_mean_accuracy", "mean"),
                seeds=("seed", "nunique"),
            )
            .sort_values(["dataset", "budget_labels"])
        )
    else:
        remaining_summary = pd.DataFrame()
    remaining_summary.to_csv(outdir / "remaining_degree_at_best_setting_mean.csv", index=False)
    plot_remaining_degree(remaining_summary, plots_dir)

    separability = class_separability(selection=selection, datasets=datasets, device=args.device)
    separability.to_csv(outdir / "dataset_class_separability.csv", index=False)

    final_budget = selection.groupby("dataset")["budget_labels"].transform("max")
    final = selection[selection["budget_labels"] == final_budget].copy()
    best_final = (
        final.groupby(["dataset", "setting"], as_index=False)
        .agg(final_accuracy=("test_accuracy_after_episode", "mean"))
        .sort_values(["dataset", "final_accuracy"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .head(1)
    )
    density_final = (
        density[["dataset", "setting", "effective_degree_mean", "sigma_mean", "sigma_cv"]]
        .groupby(["dataset", "setting"], as_index=False)
        .mean(numeric_only=True)
    )
    cross = best_final.merge(density_final, on=["dataset", "setting"], how="left").merge(
        separability, on="dataset", how="left"
    )
    cross = cross.rename(columns={"effective_degree_mean": "best_final_effective_degree"})
    cross.to_csv(outdir / "dataset_cross_summary.csv", index=False)
    plot_dataset_cross_summary(cross, plots_dir)

    print(f"Wrote {outdir / 'best_setting_by_budget.csv'}")
    print(f"Wrote {outdir / 'remaining_degree_at_best_setting_by_seed.csv'}")
    print(f"Wrote {outdir / 'remaining_degree_at_best_setting_mean.csv'}")
    print(f"Wrote {outdir / 'dataset_class_separability.csv'}")
    print(f"Wrote {outdir / 'dataset_cross_summary.csv'}")
    print(f"Wrote plots under {plots_dir}")


if __name__ == "__main__":
    main()
