"""
Scanner: discover and aggregate all result.json files under output/.

Usage
-----
    # In Python:
    from reporting.scanner import load_all_results, get_sweeps_summary, get_all_aggregated
    df = load_all_results()

    # CLI (print summary):
    python reporting/scanner.py
    python reporting/scanner.py --output_dir /path/to/output
    python reporting/scanner.py --save results.parquet
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# Keys excluded when hashing params → so runs with different seeds are grouped.
_EXCLUDE_FROM_KEY = {
    "seed",
    "exp_name",
    "exp-name",
}

# Display order for human-readable config labels.
_LABEL_KEY_ORDER = [
    "al_method",
    "eval_model_type",
    "dataset",
    "model",
    "budget",
    "max_iter",
    "tau",
    "target_nu",
    "initial_delta",
    "sparse_K",
    "kernel_type",
    "diff_method",
    "alpha",
    "features_tag",
]

# Args that can be extracted from the command string as a fallback when a
# result.json was written before the param was added to ResultWriter.
_COMMAND_BACKFILL: dict[str, type] = {
    "tau": float,
    "target_nu": float,
}


# ── Config key / label ────────────────────────────────────────────────────────


def compute_config_key(params: dict) -> str:
    """
    Return a 12-char MD5 hex string that uniquely identifies a configuration,
    ignoring seed and experiment-name fields.  Used for grouping across seeds.
    """
    filtered = {k: v for k, v in params.items() if k not in _EXCLUDE_FROM_KEY}
    key_str = json.dumps(filtered, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def compute_config_label(params: dict) -> str:
    """Short, human-readable description of a configuration."""
    parts = []
    for k in _LABEL_KEY_ORDER:
        v = params.get(k)
        if v is None or v == -1:
            continue
        if v is False:
            continue
        if v is True:
            parts.append(k)
        else:
            parts.append(f"{k}={v}")
    # Append any extra params not in the standard order
    seen = set(_LABEL_KEY_ORDER) | _EXCLUDE_FROM_KEY
    for k, v in sorted(params.items()):
        if k in seen or v is None or v == -1 or v is False:
            continue
        parts.append(f"{k}={v}" if v is not True else k)
    return " | ".join(parts) if parts else "unknown"


# ── Loading ───────────────────────────────────────────────────────────────────


def load_all_results(output_root=None) -> pd.DataFrame:
    """
    Recursively find every result.json under *output_root* and return a
    DataFrame with one row per run.

    Param columns are prefixed with ``p_`` (e.g. ``p_seed``, ``p_al_method``).
    The ``episodes`` column holds the raw list-of-dicts for per-episode data.
    """
    output_root = Path(output_root) if output_root else REPO_ROOT / "output"
    records = []

    for result_file in sorted(output_root.rglob("result.json")):
        try:
            with open(result_file) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        params = dict(data.get("params") or {})
        # Back-fill params that old result.json files may be missing by parsing
        # them out of the saved command string.
        cmd = data.get("command", "")
        for _arg, _cast in _COMMAND_BACKFILL.items():
            if params.get(_arg) is None and f"--{_arg}" in cmd:
                import re as _re
                _m = _re.search(rf"--{_arg}\s+([\S]+)", cmd)
                if _m:
                    try:
                        params[_arg] = _cast(_m.group(1))
                    except (ValueError, TypeError):
                        pass
        record = {
            "exp_dir": data.get("exp_dir", str(result_file.parent)),
            "result_file": str(result_file),
            "status": data.get("status", "unknown"),
            "created_at": data.get("created_at"),
            "finished_at": data.get("finished_at"),
            "sweep_id": data.get("sweep_id", "unknown"),
            "sweep_name": data.get("sweep_name") or data.get("sweep_id", "unknown"),
            "slurm_task_id": data.get("slurm_task_id", "0"),
            "commit_hash": data.get("commit_hash", "unknown"),
            "command": data.get("command", ""),
            "config_key": compute_config_key(params),
            "config_label": compute_config_label(params),
            "total_episodes_planned": data.get("total_episodes_planned", 0),
            "total_episodes_done": data.get("total_episodes_done", 0),
            "final_test_acc": data.get("final_test_acc"),
            "episodes": data.get("episodes", []),
            "error": data.get("error"),
            **{f"p_{k}": v for k, v in params.items()},
        }
        records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    for col in ("created_at", "finished_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df.sort_values("created_at", ascending=False).reset_index(drop=True)


# ── Sweep summary ─────────────────────────────────────────────────────────────


def get_sweeps_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (sweep_id, sweep_name) with status counts and timing."""
    if df.empty:
        return pd.DataFrame()

    rows = []
    for (sweep_id, sweep_name), g in df.groupby(["sweep_id", "sweep_name"], sort=False):
        counts = g["status"].value_counts().to_dict()
        rows.append(
            {
                "sweep_id": sweep_id,
                "sweep_name": sweep_name,
                "total": len(g),
                "completed": counts.get("completed", 0),
                "running": counts.get("running", 0),
                "failed": counts.get("failed", 0),
                "created_at": g["created_at"].min(),
                "commit_hash": g["commit_hash"].iloc[0],
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("created_at", ascending=False)
        .reset_index(drop=True)
    )


# ── Per-config aggregation ────────────────────────────────────────────────────


def aggregate_config_episodes(group_df: pd.DataFrame) -> dict:
    """
    Given a group of runs sharing (sweep_id, config_key), compute mean ± std
    of per-episode test_acc across seeds.

    Returns a dict with keys:
        config_label, config_key, sweep_name, n_seeds,
        agg (DataFrame with episode/test_acc_mean/test_acc_std/val_acc_mean/labeled_size/n_seeds),
        final_test_acc_mean, final_test_acc_std
    """
    episode_rows = []
    for _, row in group_df.iterrows():
        seed = row.get("p_seed")
        for ep in row.get("episodes", []):
            episode_rows.append(
                {
                    "episode": ep.get("episode"),
                    "test_acc": ep.get("test_acc"),
                    "val_acc": ep.get("val_acc"),
                    "labeled_size": ep.get("labeled_size"),
                    "seed": seed,
                }
            )

    base = {
        "config_label": group_df.iloc[0]["config_label"],
        "config_key": group_df.iloc[0]["config_key"],
        "sweep_name": group_df.iloc[0]["sweep_name"],
    }

    if not episode_rows:
        return {**base, "n_seeds": 0, "agg": pd.DataFrame(),
                "final_test_acc_mean": None, "final_test_acc_std": None}

    ep_df = pd.DataFrame(episode_rows).dropna(subset=["episode", "test_acc"])
    if ep_df.empty:
        return {**base, "n_seeds": 0, "agg": pd.DataFrame(),
                "final_test_acc_mean": None, "final_test_acc_std": None}

    agg = (
        ep_df.groupby("episode")
        .agg(
            test_acc_mean=("test_acc", "mean"),
            test_acc_std=("test_acc", "std"),
            val_acc_mean=("val_acc", "mean"),
            labeled_size=("labeled_size", "first"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    agg["test_acc_std"] = agg["test_acc_std"].fillna(0.0)

    last = agg.iloc[-1]
    return {
        **base,
        "n_seeds": int(agg["n_seeds"].max()),
        "agg": agg,
        "final_test_acc_mean": float(last["test_acc_mean"]),
        "final_test_acc_std": float(last["test_acc_std"]),
    }


def get_all_aggregated(df: pd.DataFrame) -> list:
    """Return list of aggregated dicts, one per (sweep_id, config_key) group."""
    if df.empty:
        return []
    return [
        aggregate_config_episodes(g)
        for _, g in df.groupby(["sweep_id", "config_key"], sort=False)
    ]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan output/ for result.json files")
    parser.add_argument(
        "--output_dir", default=None,
        help="Root output directory (default: <repo>/output/)"
    )
    parser.add_argument(
        "--save", default=None,
        help="Save DataFrame to this .parquet path"
    )
    cli = parser.parse_args()

    results_df = load_all_results(cli.output_dir)
    if results_df.empty:
        print("No result.json files found.")
    else:
        print(f"Found {len(results_df)} runs.\n")
        summary = get_sweeps_summary(results_df)
        print("=== Sweeps ===")
        print(summary.to_string(index=False))

        show_cols = [
            c for c in
            ["sweep_name", "config_label", "p_seed", "status",
             "total_episodes_done", "total_episodes_planned", "final_test_acc"]
            if c in results_df.columns
        ]
        print("\n=== Runs ===")
        print(results_df[show_cols].to_string(index=False))

        if cli.save:
            save_df = results_df.copy()
            save_df["episodes"] = save_df["episodes"].apply(json.dumps)
            save_df.to_parquet(cli.save)
            print(f"\nSaved to {cli.save}")
