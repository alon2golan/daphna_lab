"""
plot_repro_4_15_B300_compare.py
-------------------------------
Compare the noisy AL curves under two ν′ calibrations, against the oracle:

  oracle        : τ=1.0 run (no noise; from the B100-ν′ repro sweep)
  noisy B100-ν′ : repro4_15_full_*   (ν′ looked up at B=100)
  noisy B300-ν′ : repro4_15_B300nu_* (ν′ estimated at B=300, lower → less noise)

All trained at --budget 100, 11 episodes, seed-avg(1,2,3).

Figure: 2 rows (probcover, random) × 3 cols (τ=0.2,0.5,0.8). Each panel:
  oracle (blue solid), noisy-B100 (orange dashed), noisy-B300 (green solid).
Also prints a three-way ep10 gap table.

Out:
  results/cifar100/repro_4_15/repro_4_15_B300_compare.png
  results/cifar100/repro_4_15/B300_compare_gap_table.csv
"""
from __future__ import annotations

import csv
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "results" / "cifar100" / "repro_4_15"
NOISY_TAUS = [0.2, 0.5, 0.8]
STRATS = ["probcover", "random"]


def load_sweep(prefix: str):
    files = [l.strip() for l in os.popen(
        f"grep -rl '{prefix}' --include=result.json {REPO}/output/ 2>/dev/null")]
    rows = []
    names = set()
    for f in files:
        d = json.load(open(f))
        sn = d.get("sweep_name", "")
        if not sn.startswith(prefix):
            continue
        names.add(sn)
        cmd = d["command"]
        strat = "probcover" if "alprobcover" in cmd else "random"
        m = re.search(r"--tau (\S+)", cmd)
        tau = float(m.group(1)) if m else 1.0
        seed = int(re.search(r"--seed (\d+)", cmd).group(1))
        for e in d.get("episodes", []):
            rows.append((sn, strat, tau, seed, e["episode"], e["test_acc"]))
    if names:
        latest = sorted(names)[-1]
        rows = [r for r in rows if r[0] == latest]
    return rows


def seed_avg(rows):
    acc = defaultdict(list)
    for _, strat, tau, seed, ep, ta in rows:
        acc[(strat, tau, ep)].append(ta)
    return {k: statistics.mean(v) for k, v in acc.items()}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    b100 = seed_avg(load_sweep("repro4_15_full_"))
    b300 = seed_avg(load_sweep("repro4_15_B300nu_"))
    if not b300:
        print("B=300 rerun not in yet — nothing to plot.")
        return
    print(f"b100 keys={len(b100)} b300 keys={len(b300)}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey="row")
    for ri, strat in enumerate(STRATS):
        for ci, tau in enumerate(NOISY_TAUS):
            ax = axes[ri][ci]
            eps = sorted({k[2] for k in b100 if k[0] == strat and k[1] == 1.0})
            # oracle (from b100 sweep; identical run regardless of nu')
            orc = [b100[(strat, 1.0, e)] for e in eps]
            ax.plot(eps, orc, "-", color="#1f77b4", lw=2, marker="o", ms=3, label="oracle (τ=1)")
            n1 = [b100.get((strat, tau, e)) for e in eps]
            ax.plot(eps, n1, "--", color="#ff7f0e", lw=1.8, marker="s", ms=3, label="noisy · B100-ν′")
            n3 = [b300.get((strat, tau, e)) for e in eps]
            ax.plot(eps, n3, "-", color="#2ca02c", lw=1.8, marker="^", ms=3, label="noisy · B300-ν′")
            ax.set_title(f"{strat} · τ={tau}", fontsize=10)
            ax.grid(alpha=0.3)
            if ri == 1:
                ax.set_xlabel("AL episode")
            if ci == 0:
                ax.set_ylabel(f"{strat}\ntest acc (%)")
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("4_15 noisy AL: B=100 ν′ vs B=300 ν′ (lower noise) vs oracle — CIFAR100 · trained at B=100 · seed-avg(1,2,3)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUTDIR / "repro_4_15_B300_compare.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)

    # three-way ep10 gap table
    rows = []
    for strat in STRATS:
        orc = b100.get((strat, 1.0, 10))
        for tau in NOISY_TAUS:
            n1 = b100.get((strat, tau, 10))
            n3 = b300.get((strat, tau, 10))
            rows.append(dict(strategy=strat, tau=tau, oracle=round(orc, 2),
                             noisy_B100=round(n1, 2), gap_B100=round(orc - n1, 2),
                             noisy_B300=round(n3, 2), gap_B300=round(orc - n3, 2),
                             delta=round(n3 - n1, 2)))
    with open(OUTDIR / "B300_compare_gap_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\n=== ep10 seed-avg: B100-ν′ vs B300-ν′ (gap = oracle - noisy; neg => noisy beats oracle) ===")
    print(f"{'strat':10s}{'τ':>5s} | {'oracle':>7s} {'noi100':>7s} {'gap100':>7s} | "
          f"{'noi300':>7s} {'gap300':>7s} | {'Δacc':>6s}")
    for r in rows:
        print(f"{r['strategy']:10s}{r['tau']:>5.1f} | {r['oracle']:>7} {r['noisy_B100']:>7} {r['gap_B100']:>7} | "
              f"{r['noisy_B300']:>7} {r['gap_B300']:>7} | {r['delta']:>+6}")


if __name__ == "__main__":
    main()
