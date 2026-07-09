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
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# Keys excluded when hashing params → so runs with different seeds are grouped.
_EXCLUDE_FROM_KEY = {
    "seed",
    "exp_name",
    "exp-name",
    # NOTE: nu_tag is INTENTIONALLY part of config_key. It encodes the nu'-cache
    # provenance (<opt>_b<budget>) and is the ONLY discriminator between runs
    # whose nu_prime coincides across cache budgets — e.g. random B100 tau0.1:
    # nucache100 and nucache300 both give nu'=0.785156, so without nu_tag in the
    # key they collapse into one config and one tag (sgd_b100/sgd_b300) is lost.
    # nu_tag is constant across seeds, so it still groups seeds together.
    # (Including it changes config_key hashes → older about=<hash> URLs break;
    # accepted as the cost of correct grouping.)
    # Routing/visibility flag (dashboard Comparison vs Additional data), not a
    # configuration dimension — excluded so it neither fragments seed-grouping
    # nor clutters config labels.
    "track_progress",
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
    "nu_tag",
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

# Free-form --tag keys that are really aliases of a canonical param. Folded onto
# that param at load time so a sweep tagged `--tag noise=0.5` doesn't sprout a
# separate p_noise column (string "0.5") that duplicates the real p_noise_rate
# param (float 0.5) but only for that sweep — which would fragment the dashboard's
# "noise" filter/facet and silently drop every sweep lacking the tag from a
# cross-sweep comparison. {tag_key: canonical_param}.
_TAG_ALIASES: dict[str, str] = {
    "noise": "noise_rate",
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
        # When the run carries a nu_tag (rendered above), it conveys the
        # provenance of nu_prime far more readably than the raw float, so drop the
        # redundant nu_prime from the compact legend (except the oracle "none" tag,
        # which has no nu_prime to drop anyway).
        if k == "nu_prime" and params.get("nu_tag") not in (None, "none"):
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

    # result.json always sits at <dataset>/<model>/<date>/<run>/result.json.
    # A fixed-depth glob avoids descending into the ~thousands of episode_*/
    # subdirs (.npy/.pyth checkpoints) that rglob would needlessly stat-walk —
    # ~100x faster on NFS (1.9s -> 0.02s) for an identical result set.
    for result_file in sorted(output_root.glob("*/*/*/*/result.json")):
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
        # Free-form run tags recorded by --tag KEY=VALUE (result.json["tags"]).
        # Exposed below as p_<key> facet columns. nu_tag, if explicitly tagged,
        # also feeds the config label / nu_prime-suppression logic, so it is
        # mirrored into params here (an explicit tag wins over the derivation).
        tags = dict(data.get("tags") or {})
        if "nu_tag" in tags:
            params["nu_tag"] = tags["nu_tag"]

        # Fold aliased tags onto their canonical param (see _TAG_ALIASES) so the
        # dashboard offers ONE noise column shared by every sweep. When the
        # canonical param is already recorded (the usual case), the redundant tag
        # is simply dropped — leaving config_key / about= hashes untouched; when
        # it's absent, the tag value fills it (numeric when parseable).
        for _tk, _canon in _TAG_ALIASES.items():
            if _tk not in tags:
                continue
            if params.get(_canon) is None:
                _tv = tags[_tk]
                try:
                    params[_canon] = float(_tv)
                except (TypeError, ValueError):
                    params[_canon] = _tv
            tags.pop(_tk, None)

        # Provenance of the nu' (injected noise level) this run used, as a single
        # readable tag. Used when the run carries no explicit nu_tag. Derived from
        # the exp-name / command, display only, so it is excluded from config_key
        # (about=<hash> URLs stay stable). Values:
        #   "none"            oracle arm (target_nu=0, no injected noise)
        #   "nu_analytical"   analytical rule nu'=tau (exp-name "...analytical...")
        #   "<opt>_b<N>"      NoAA-estimated nu' from the <opt> nu_prime cache at
        #                     budget N (exp-name "...nucache<N>..."); <opt> is sgd
        #                     when the run is from an sgd sweep, else adam. AL batch
        #                     and the nu'-cache budget are decoupled.
        if params.get("nu_tag") is None:
            _hay = (data.get("command", "") or "") + " " + str(result_file.parent)
            _tgt = params.get("target_nu")
            try:
                _is_oracle = _tgt is not None and float(_tgt) == 0.0
            except (TypeError, ValueError):
                _is_oracle = False
            if "_oracle" in _hay or _is_oracle:
                params["nu_tag"] = "none"
            elif "analytical" in _hay:
                params["nu_tag"] = "nu_analytical"
            else:
                _nm = re.search(r"nucache(\d+)", _hay)
                if _nm:
                    _opt = "sgd" if "sgd" in _hay.lower() else "adam"
                    params["nu_tag"] = f"{_opt}_b{_nm.group(1)}"
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
            # Free-form tags as facet columns. config_key/label were computed from
            # params only, so tags never fragment seed-grouping or change about=
            # URLs. Skip any tag whose key already came from a real param.
            **{f"p_{k}": v for k, v in tags.items() if k not in params},
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


def _resolve_tau(group_df: pd.DataFrame) -> float:
    """
    The effective annotator τ for a config, defaulting to 1.0 (oracle / clean).
    Used to derive the oracle-cost axis: budget_units = labeled_size * τ.
    """
    tau = group_df.iloc[0].get("p_tau")
    try:
        tau = float(tau)
    except (TypeError, ValueError):
        return 1.0
    return tau if tau and tau > 0 else 1.0


def aggregate_config_episodes(group_df: pd.DataFrame) -> dict:
    """
    Given a group of runs sharing (sweep_id, config_key), compute mean ± std of
    EVERY numeric per-episode field across seeds (test/val/train acc, duration, …).

    Returns a dict with keys:
        config_label, config_key, sweep_name, n_seeds, tau,
        agg (DataFrame with: episode, <field>_mean/<field>_std for each numeric
             per-episode field, labeled_size [#examples], budget_units
             [=labeled_size*τ, oracle cost], n_seeds),
        final_test_acc_mean, final_test_acc_std

    The aggregator is generic so newly-logged per-episode metrics surface
    automatically (Feature 2's Y-axis dropdown discovers the *_mean columns).
    """
    episode_rows = []
    for _, row in group_df.iterrows():
        seed = row.get("p_seed")
        for ep in row.get("episodes", []):
            if not isinstance(ep, dict):
                continue
            rec = {k: v for k, v in ep.items()}
            rec["seed"] = seed
            episode_rows.append(rec)

    base = {
        "config_label": group_df.iloc[0]["config_label"],
        "config_key": group_df.iloc[0]["config_key"],
        "sweep_name": group_df.iloc[0]["sweep_name"],
        "sweep_id": str(group_df.iloc[0].get("sweep_id", "")),
        "tau": _resolve_tau(group_df),
    }

    if not episode_rows:
        return {**base, "n_seeds": 0, "agg": pd.DataFrame(),
                "final_test_acc_mean": None, "final_test_acc_std": None}

    ep_df = pd.DataFrame(episode_rows).dropna(subset=["episode", "test_acc"])
    if ep_df.empty:
        return {**base, "n_seeds": 0, "agg": pd.DataFrame(),
                "final_test_acc_mean": None, "final_test_acc_std": None}

    # Every numeric per-episode field gets mean/std; labeled_size is kept as a
    # single deterministic value (#examples) rather than averaged, and the group
    # keys (episode, seed) are excluded.
    numeric_fields = [
        c for c in ep_df.columns
        if c not in ("episode", "seed", "labeled_size")
        and pd.api.types.is_numeric_dtype(ep_df[c])
    ]
    agg_spec = {}
    for f in numeric_fields:
        agg_spec[f"{f}_mean"] = (f, "mean")
        agg_spec[f"{f}_std"] = (f, "std")
    if "labeled_size" in ep_df.columns:
        agg_spec["labeled_size"] = ("labeled_size", "first")
    agg_spec["n_seeds"] = ("seed", "nunique")

    agg = ep_df.groupby("episode").agg(**agg_spec).reset_index()
    for c in agg.columns:
        if c.endswith("_std"):
            agg[c] = agg[c].fillna(0.0)

    # Derived oracle-cost axis: budget units consumed = #examples * τ
    # (oracle τ=1 → budget_units == #examples; noisy τ<1 → fewer units per example).
    if "labeled_size" in agg.columns:
        agg["budget_units"] = agg["labeled_size"] * base["tau"]

    last = agg.iloc[-1]
    return {
        **base,
        "n_seeds": int(agg["n_seeds"].max()),
        "agg": agg,
        "final_test_acc_mean": float(last["test_acc_mean"]),
        "final_test_acc_std": float(last.get("test_acc_std", 0.0)),
    }


def get_all_aggregated(df: pd.DataFrame) -> list:
    """Return list of aggregated dicts, one per (sweep_id, config_key) group."""
    if df.empty:
        return []
    return [
        aggregate_config_episodes(g)
        for _, g in df.groupby(["sweep_id", "config_key"], sort=False)
    ]


# ── Sweep auto-facts (Feature 0: description cards) ────────────────────────────

# Params surfaced in a sweep's auto-fact grid (varying ones highlighted).
_DESCRIBE_PARAM_KEYS = [
    "p_al_method", "p_tau", "p_target_nu", "p_nu_prime", "p_nu_optimizer", "p_nu_tag", "p_budget", "p_max_iter",
    "p_dataset", "p_model", "p_eval_model_type", "p_features_tag",
    "p_initial_delta", "p_kernel_type", "p_diff_method", "p_alpha",
    "p_sparse_K", "p_seed",
]


def describe_sweep(df_subset: pd.DataFrame) -> dict:
    """
    Compute the live auto-facts for a sweep (or any run subset) — never stored,
    always fresh. Reused by the Ledger card and the Graphs "About this curve"
    panel (Feature 0).

    Returns: n_runs, date_min/date_max, status_counts, varying {param: [vals]},
             constant {param: val}, command, commit_hash, exp_dir, sweep_names.
    """
    if df_subset is None or df_subset.empty:
        return {}

    dates = (
        df_subset["created_at"].dropna()
        if "created_at" in df_subset.columns else pd.Series([], dtype="datetime64[ns]")
    )
    status_counts = (
        df_subset["status"].value_counts().to_dict()
        if "status" in df_subset.columns else {}
    )

    varying, constant = {}, {}
    for c in _DESCRIBE_PARAM_KEYS:
        if c not in df_subset.columns:
            continue
        vals = sorted(df_subset[c].dropna().unique().tolist(), key=str)
        if not vals:
            continue
        if len(vals) > 1:
            varying[c[2:]] = vals
        else:
            constant[c[2:]] = vals[0]

    # Representative run = most recent (best command/commit to show).
    if "created_at" in df_subset.columns and df_subset["created_at"].notna().any():
        rep = df_subset.sort_values("created_at").iloc[-1]
    else:
        rep = df_subset.iloc[0]

    return {
        "n_runs": int(len(df_subset)),
        "date_min": dates.min() if not dates.empty else None,
        "date_max": dates.max() if not dates.empty else None,
        "status_counts": status_counts,
        "varying": varying,
        "constant": constant,
        "command": rep.get("command", ""),
        "commit_hash": rep.get("commit_hash", ""),
        "exp_dir": rep.get("exp_dir", ""),
        "sweep_names": sorted(df_subset["sweep_name"].dropna().unique().tolist())
        if "sweep_name" in df_subset.columns else [],
    }


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
