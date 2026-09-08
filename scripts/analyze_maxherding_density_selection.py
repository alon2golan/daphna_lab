#!/usr/bin/env python3
"""
Retrospective density and selection diagnostics for MaxHerding runs.

This script does not train models or change experiment outputs. It reads saved
active-learning run folders, recomputes the same feature-space distance
quantities used by the kernel, and writes CSV summaries that help answer:

  * How dense is the graph induced by a given delta / local sigma K?
  * Are selected points in denser or sparser regions than a random anchor set?
  * How do selected-point density and sigma change over AL episodes?

The expensive part is density estimation. By default, dataset-level density is
estimated on a sampled anchor set, while selected-point density is exact against
all training points.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "deep_al") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "deep_al"))

from pycls.al.kernel_utils import compute_local_sigmas  # noqa: E402
from pycls.datasets.utils.features import load_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze density and selected points in saved MaxHerding runs."
    )
    parser.add_argument(
        "--output-root",
        default="output",
        help="Root directory containing saved AL runs.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=None,
        help="Specific run directory to analyze. Can be passed multiple times.",
    )
    parser.add_argument(
        "--run-glob",
        default="output/*/resnet18/2026_4_23/maxherding_cutoff_*",
        help="Glob used to discover run directories when --run-dir is not provided.",
    )
    parser.add_argument(
        "--name-regex",
        default=r"maxherding_cutoff_.*_(global_sigma|local_sigma_k\d+)_",
        help="Regex matched against run directory names.",
    )
    parser.add_argument(
        "--outdir",
        default="results_summaries/maxherding_density_selection",
        help="Directory for CSV outputs and local sigma cache.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        choices=["CIFAR10", "CIFAR100", "TINYIMAGENET"],
        help="Optional dataset filter.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Optional RNG seed filter.",
    )
    parser.add_argument(
        "--local-k",
        nargs="*",
        type=int,
        default=None,
        help="Optional local sigma K filter. Global runs are kept unless --local-only is set.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Drop global-sigma runs.",
    )
    parser.add_argument(
        "--global-only",
        action="store_true",
        help="Drop local-sigma runs.",
    )
    parser.add_argument(
        "--anchor-sample-size",
        type=int,
        default=5000,
        help="Number of dataset anchor points for density summaries.",
    )
    parser.add_argument(
        "--anchor-seed",
        type=int,
        default=0,
        help="Random seed for sampled dataset anchors.",
    )
    parser.add_argument(
        "--skip-dataset-density",
        action="store_true",
        help="Only compute exact selected-point diagnostics.",
    )
    parser.add_argument(
        "--skip-selected-density",
        action="store_true",
        help="Skip selected-point distance counts. Useful for fast metadata smoke tests.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for distance computations.",
    )
    parser.add_argument(
        "--distance-batch-size",
        type=int,
        default=128,
        help="Anchor batch size for cdist when counting neighbors.",
    )
    parser.add_argument(
        "--sigma-batch-size",
        type=int,
        default=512,
        help="Batch size when recomputing local sigma vectors.",
    )
    return parser.parse_args()


def read_config(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def discover_runs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        return sorted(Path(path) for path in args.run_dir)

    pattern = re.compile(args.name_regex)
    runs = []
    for run_dir in Path(".").glob(args.run_glob):
        if (run_dir / "config.yaml").exists() and pattern.search(run_dir.name):
            runs.append(run_dir)
    return sorted(runs)


def get_config_value(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def run_metadata(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    dataset = get_config_value(config, "DATASET.NAME")
    seed = int(get_config_value(config, "RNG_SEED", -1))
    local_sigma = bool(
        get_config_value(config, "LOCAL_SIGMA", get_config_value(config, "ACTIVE_LEARNING.LOCAL_SIGMA", False))
    )
    local_k = get_config_value(
        config, "LOCAL_SIGMA_K", get_config_value(config, "ACTIVE_LEARNING.LOCAL_SIGMA_K", None)
    )
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "dataset": dataset,
        "seed": seed,
        "eval_model_type": get_config_value(
            config, "EVAL_MODEL_TYPE", get_config_value(config, "ACTIVE_LEARNING.EVAL_MODEL_TYPE")
        ),
        "delta": float(get_config_value(config, "ACTIVE_LEARNING.INITIAL_DELTA")),
        "global_sigma": float(get_config_value(config, "ACTIVE_LEARNING.INITIAL_SIGMA", 1.0)),
        "local_sigma": local_sigma,
        "local_k": int(local_k) if local_sigma and local_k is not None else "",
        "cutoff_mode": get_config_value(
            config, "RBF_CUTOFF_MODE", get_config_value(config, "ACTIVE_LEARNING.RBF_CUTOFF_MODE", "none")
        ),
        "features_normalized": bool(
            get_config_value(
                config, "FEATURES_NORMALIZED", get_config_value(config, "DATASET.FEATURES_NORMALIZED", True)
            )
        ),
        "features_tag": get_config_value(
            config, "FEATURES_TAG", get_config_value(config, "DATASET.FEATURES_TAG", None)
        ),
        "budget_size": int(get_config_value(config, "ACTIVE_LEARNING.BUDGET_SIZE", 0)),
        "max_iter": int(get_config_value(config, "ACTIVE_LEARNING.MAX_ITER", 0)),
    }


def keep_run(args: argparse.Namespace, meta: dict[str, Any]) -> bool:
    if args.datasets is not None and meta["dataset"] not in set(args.datasets):
        return False
    if args.seeds is not None and meta["seed"] not in set(args.seeds):
        return False
    if args.local_only and not meta["local_sigma"]:
        return False
    if args.global_only and meta["local_sigma"]:
        return False
    if args.local_k is not None and meta["local_sigma"]:
        if int(meta["local_k"]) not in set(args.local_k):
            return False
    return True


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
        raise ValueError(f"Unsupported dataset for label loading: {dataset}")
    return np.asarray(ds.targets, dtype=np.int64)


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_cv": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_q05": math.nan,
            f"{prefix}_q25": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_q75": math.nan,
            f"{prefix}_q95": math.nan,
            f"{prefix}_max": math.nan,
        }

    mean = float(np.mean(values))
    std = float(np.std(values))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_cv": float(std / mean) if mean else math.nan,
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_q05": float(np.quantile(values, 0.05)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_q95": float(np.quantile(values, 0.95)),
        f"{prefix}_max": float(np.max(values)),
    }


def label_entropy(labels: np.ndarray, num_classes: int) -> tuple[float, float]:
    if labels.size == 0:
        return math.nan, math.nan
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float64)
    probs = counts[counts > 0] / labels.size
    entropy = float(-np.sum(probs * np.log(probs)))
    normalized = entropy / math.log(num_classes) if num_classes > 1 else math.nan
    return entropy, normalized


def cache_key(dataset: str, normalized: bool, features_tag: str | None, local_k: int) -> str:
    tag = features_tag or "default"
    norm = "norm" if normalized else "unnorm"
    return f"{dataset}_{norm}_{tag}_k{local_k}".replace("/", "_")


def get_local_sigmas(
    *,
    features: np.ndarray,
    dataset: str,
    normalized: bool,
    features_tag: str | None,
    local_k: int,
    outdir: Path,
    device: str,
    batch_size: int,
) -> np.ndarray:
    cache_dir = outdir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"local_sigmas_{cache_key(dataset, normalized, features_tag, local_k)}.npy"
    if cache_path.exists():
        return np.load(cache_path)

    sigmas = compute_local_sigmas(
        features,
        k=local_k,
        batch_size=batch_size,
        device=device,
        dtype=torch.float32,
    ).numpy()
    np.save(cache_path, sigmas)
    return sigmas


def count_neighbors(
    *,
    features: np.ndarray,
    anchor_indices: np.ndarray,
    delta: float,
    global_sigma: float,
    local_sigmas: np.ndarray | None,
    device: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
    raw_counts = []
    effective_counts = []
    raw_threshold = float(delta)
    delta_sq = float(delta) ** 2
    sigma_all = None
    if local_sigmas is not None:
        sigma_all = torch.as_tensor(local_sigmas, dtype=torch.float32, device=device)

    with torch.no_grad():
        for start in range(0, len(anchor_indices), batch_size):
            batch_idx_np = anchor_indices[start : start + batch_size]
            batch_idx = torch.as_tensor(batch_idx_np, dtype=torch.long, device=device)
            anchors = feature_tensor[batch_idx]
            dist = torch.cdist(anchors, feature_tensor)

            raw = torch.count_nonzero(dist < raw_threshold, dim=1)
            raw = raw - 1  # remove self-neighbor
            raw_counts.append(raw.detach().cpu().numpy().astype(np.int64))

            if sigma_all is None:
                scaled_dist = dist / float(global_sigma)
                effective = torch.count_nonzero(scaled_dist < raw_threshold, dim=1)
            else:
                sigma_i = sigma_all[batch_idx].unsqueeze(1)
                denom = torch.clamp(sigma_i * sigma_all.unsqueeze(0), min=1e-12)
                effective = torch.count_nonzero((dist * dist) / denom < delta_sq, dim=1)
            effective = effective - 1
            effective_counts.append(effective.detach().cpu().numpy().astype(np.int64))

            del dist

    return {
        "raw_degree": np.concatenate(raw_counts) if raw_counts else np.asarray([], dtype=np.int64),
        "effective_degree": np.concatenate(effective_counts)
        if effective_counts
        else np.asarray([], dtype=np.int64),
    }


def read_accuracy(run_dir: Path) -> tuple[list[int], list[float]]:
    x_path = run_dir / "plot_episode_xvalues.npy"
    y_path = run_dir / "plot_episode_yvalues.npy"
    if not x_path.exists() or not y_path.exists():
        return [], []
    return np.load(x_path).astype(int).tolist(), np.load(y_path).astype(float).tolist()


def active_sets(run_dir: Path) -> list[tuple[int, np.ndarray]]:
    result = []
    for episode_dir in sorted(run_dir.glob("episode_*"), key=lambda p: int(p.name.split("_")[1])):
        active_path = episode_dir / "activeSet.npy"
        if active_path.exists():
            result.append((int(episode_dir.name.split("_")[1]), np.load(active_path).astype(np.int64)))
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    feature_cache: dict[tuple[str, bool, str | None], np.ndarray] = {}
    labels_cache: dict[tuple[str, str], np.ndarray] = {}
    density_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []

    candidates = discover_runs(args)
    processed = 0
    for run_dir in candidates:
        config = read_config(run_dir)
        meta = run_metadata(run_dir, config)
        if not keep_run(args, meta):
            continue
        processed += 1
        if args.max_runs is not None and processed > args.max_runs:
            break

        dataset = meta["dataset"]
        data_root = get_config_value(config, "DATASET.ROOT_DIR")
        labels_key = (dataset, data_root)
        if labels_key not in labels_cache:
            labels_cache[labels_key] = load_labels(dataset, data_root)
        labels = labels_cache[labels_key]
        num_classes = int(np.max(labels)) + 1

        need_features = (
            not args.skip_dataset_density
            or not args.skip_selected_density
            or bool(meta["local_sigma"])
        )
        features = None
        if need_features:
            feature_key = (dataset, meta["features_normalized"], meta["features_tag"])
            if feature_key not in feature_cache:
                feature_cache[feature_key] = load_features(
                    dataset,
                    train=True,
                    normalized=meta["features_normalized"],
                    features_tag=meta["features_tag"],
                ).astype(np.float32, copy=False)
            features = feature_cache[feature_key]

        sigmas = None
        if meta["local_sigma"]:
            sigmas = get_local_sigmas(
                features=features,
                dataset=dataset,
                normalized=meta["features_normalized"],
                features_tag=meta["features_tag"],
                local_k=int(meta["local_k"]),
                outdir=outdir,
                device=args.device,
                batch_size=args.sigma_batch_size,
            )

        if not args.skip_dataset_density:
            rng = np.random.default_rng(args.anchor_seed)
            anchor_count = min(args.anchor_sample_size, len(features))
            anchors = np.sort(rng.choice(len(features), size=anchor_count, replace=False))
            degrees = count_neighbors(
                features=features,
                anchor_indices=anchors,
                delta=meta["delta"],
                global_sigma=meta["global_sigma"],
                local_sigmas=sigmas,
                device=args.device,
                batch_size=args.distance_batch_size,
            )
            density_row = dict(meta)
            density_row.update({"anchor_count": anchor_count})
            density_row.update(summarize(degrees["raw_degree"], "raw_degree"))
            density_row.update(summarize(degrees["effective_degree"], "effective_degree"))
            if sigmas is not None:
                density_row.update(summarize(sigmas, "sigma"))
            density_rows.append(density_row)

        x_values, y_values = read_accuracy(run_dir)
        accuracy_by_episode = {i: y for i, y in enumerate(y_values)}
        labels_by_episode = {i: x for i, x in enumerate(x_values)}

        for episode, active in active_sets(run_dir):
            selected_labels = labels[active]
            if args.skip_selected_density:
                selected_degrees = {
                    "raw_degree": np.full(len(active), np.nan),
                    "effective_degree": np.full(len(active), np.nan),
                }
            else:
                selected_degrees = count_neighbors(
                    features=features,
                    anchor_indices=active,
                    delta=meta["delta"],
                    global_sigma=meta["global_sigma"],
                    local_sigmas=sigmas,
                    device=args.device,
                    batch_size=args.distance_batch_size,
                )
            entropy, normalized_entropy = label_entropy(selected_labels, num_classes)
            counts = Counter(selected_labels.tolist())
            episode_row = dict(meta)
            episode_row.update(
                {
                    "episode": episode,
                    "num_selected": int(len(active)),
                    "labels_after_episode": labels_by_episode.get(episode, ""),
                    "test_accuracy_after_episode": accuracy_by_episode.get(episode, ""),
                    "unique_selected_classes": int(len(counts)),
                    "selected_label_entropy": entropy,
                    "selected_label_entropy_normalized": normalized_entropy,
                    "majority_class_fraction": float(max(counts.values()) / len(active))
                    if len(active)
                    else math.nan,
                    "class_counts_json": json.dumps(dict(sorted(counts.items()))),
                }
            )
            episode_row.update(summarize(selected_degrees["raw_degree"], "selected_raw_degree"))
            episode_row.update(
                summarize(selected_degrees["effective_degree"], "selected_effective_degree")
            )
            if sigmas is not None:
                episode_row.update(summarize(sigmas[active], "selected_sigma"))
            episode_rows.append(episode_row)

            for local_pos, point_idx in enumerate(active.tolist()):
                row = dict(meta)
                row.update(
                    {
                        "episode": episode,
                        "position_in_batch": local_pos,
                        "point_index": int(point_idx),
                        "label": int(labels[point_idx]),
                        "raw_degree": ""
                        if np.isnan(selected_degrees["raw_degree"][local_pos])
                        else int(selected_degrees["raw_degree"][local_pos]),
                        "effective_degree": ""
                        if np.isnan(selected_degrees["effective_degree"][local_pos])
                        else int(selected_degrees["effective_degree"][local_pos]),
                        "sigma": float(sigmas[point_idx]) if sigmas is not None else "",
                    }
                )
                point_rows.append(row)

    write_csv(outdir / "density_summary.csv", density_rows)
    write_csv(outdir / "selection_episode_summary.csv", episode_rows)
    write_csv(outdir / "selected_points.csv", point_rows)
    print(f"Processed {min(processed, args.max_runs or processed)} runs")
    print(f"Wrote {outdir / 'density_summary.csv'}")
    print(f"Wrote {outdir / 'selection_episode_summary.csv'}")
    print(f"Wrote {outdir / 'selected_points.csv'}")


if __name__ == "__main__":
    main()
