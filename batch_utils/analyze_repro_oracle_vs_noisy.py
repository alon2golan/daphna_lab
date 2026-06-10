"""
analyze_repro_oracle_vs_noisy.py
--------------------------------
Compare the post-fix oracle-vs-noisy reproduction (sweep_name starts with
'repro4_15_oracle_vs_noisy') to the original pre-fix 2026_4_15 numbers, to see
whether the seed-decoupling noise fix closed the oracle>noisy gap.

Run:
    venv/bin/python batch_utils/analyze_repro_oracle_vs_noisy.py
"""
from __future__ import annotations
import json, glob, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "output" / "CIFAR100" / "resnet18"

# Pre-fix 2026_4_15 probcover seed-1 ep10 finals (from result.json), for reference.
PREFIX_SEED1 = {"oracle": 47.74, 0.8: 46.70, 0.5: 44.52, 0.2: 42.64}


def load_runs(sweep_prefix="repro4_15_oracle_vs_noisy"):
    """Return {(label, seed): [per-episode test_acc]} for the repro sweep."""
    runs = {}
    for rj in glob.glob(str(OUT / "**" / "result.json"), recursive=True):
        try:
            d = json.load(open(rj))
        except (json.JSONDecodeError, OSError):
            continue
        if not str(d.get("sweep_name", "")).startswith(sweep_prefix):
            continue
        p = d.get("params", {})
        if p.get("al_method") != "probcover":
            continue
        seed = p.get("seed")
        tau = p.get("tau")
        tnu = p.get("target_nu")
        label = "oracle" if (tnu == 0.0 or (tau is None and tnu is not None)) else tau
        accs = [e.get("test_acc") for e in d.get("episodes", [])]
        runs[(label, seed)] = accs
    return runs


def main():
    runs = load_runs()
    if not runs:
        print("No repro results found yet (sweep still pending/running).")
        return

    labels = ["oracle", 0.8, 0.5, 0.2]
    seeds = sorted({s for (_, s) in runs})
    print(f"seeds present: {seeds}\n")

    # Seed-averaged final-episode accuracy + gap vs oracle
    print("=== seed-averaged final-episode (ep10) test acc, post-fix ===")
    oracle_finals = [runs[("oracle", s)][-1] for s in seeds
                     if ("oracle", s) in runs and runs[("oracle", s)]]
    oracle_mean = float(np.mean(oracle_finals)) if oracle_finals else None
    for lab in labels:
        finals = [runs[(lab, s)][-1] for s in seeds
                  if (lab, s) in runs and runs[(lab, s)]]
        if not finals:
            print(f"  {str(lab):>7}: (no data)")
            continue
        m = float(np.mean(finals))
        gap = "" if (lab == "oracle" or oracle_mean is None) else f"  gap={oracle_mean - m:+.2f}"
        n = len(finals)
        print(f"  {str(lab):>7}: {m:6.2f}  (n={n}){gap}")

    # Before/after gap table (seed 1, ep10) vs the 4_15 baseline
    print("\n=== seed-1 ep10 gap: BEFORE (4_15) vs AFTER (this fix) ===")
    o_before = PREFIX_SEED1["oracle"]
    o_after = runs.get(("oracle", 1), [None])[-1] if ("oracle", 1) in runs else None
    print(f"{'tau':>6} | {'before gap':>10} | {'after gap':>10}")
    for lab in [0.8, 0.5, 0.2]:
        before = o_before - PREFIX_SEED1[lab]
        if ("oracle", 1) in runs and (lab, 1) in runs and o_after is not None:
            after = o_after - runs[(lab, 1)][-1]
            print(f"{lab:>6} | {before:>10.2f} | {after:>10.2f}")
        else:
            print(f"{lab:>6} | {before:>10.2f} | {'(pending)':>10}")


if __name__ == "__main__":
    main()
