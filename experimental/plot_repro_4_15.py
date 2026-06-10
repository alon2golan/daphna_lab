"""
plot_repro_4_15.py
------------------
Before/after of the 4_15 oracle-vs-noisy AL figure, for BOTH strategies.

  before = original sweep probcover_random__RESNET18__seed1to3__20260415_153802
           (buggy annotator: noise RNG coupled to RNG_SEED -> ~45% realized
            noise at seed 1; Adam nu' cache)
  after  = repro4_15_full_* (fixed annotator NOISE_SEED_OFFSET; same Adam nu')

Produces:
  results/cifar100/repro_4_15/repro_4_15_before_after.png
  results/cifar100/repro_4_15/after_curves.csv
  results/cifar100/repro_4_15/gap_table.csv   (seed-avg ep10, before vs after)

Layout: 2 rows (probcover, random) x 2 cols (before, after). Each panel plots
seed-averaged test_acc vs AL episode for tau in {1.0 oracle, 0.8, 0.5, 0.2}.
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
BEFORE_CSV = OUTDIR / "baseline_before_curves.csv"
AFTER_SWEEP_PREFIX = "repro4_15_full_"

TAUS = [1.0, 0.8, 0.5, 0.2]
TAU_LABEL = {1.0: "τ=1.0 (oracle)", 0.8: "τ=0.8", 0.5: "τ=0.5", 0.2: "τ=0.2"}
TAU_COLOR = {1.0: "#1f77b4", 0.8: "#2ca02c", 0.5: "#ff7f0e", 0.2: "#d62728"}
STRATS = ["probcover", "random"]


def load_before():
    rows = []
    with open(BEFORE_CSV) as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(strategy=r["strategy"], tau=float(r["tau"]),
                             seed=int(r["seed"]), episode=int(r["episode"]),
                             test_acc=float(r["test_acc"])))
    return rows


def load_after():
    """Walk output/ for the latest repro4_15_full_* sweep result.json files."""
    files = [l.strip() for l in os.popen(
        f"grep -rl '{AFTER_SWEEP_PREFIX}' --include=result.json {REPO}/output/ 2>/dev/null")]
    rows = []
    sweep_names = set()
    for f in files:
        d = json.load(open(f))
        sn = d.get("sweep_name", "")
        if not sn.startswith(AFTER_SWEEP_PREFIX):
            continue
        sweep_names.add(sn)
        cmd = d["command"]
        strat = "probcover" if "alprobcover" in cmd else "random"
        m = re.search(r"--tau (\S+)", cmd)
        tau = float(m.group(1)) if m else 1.0
        seed = int(re.search(r"--seed (\d+)", cmd).group(1))
        for e in d.get("episodes", []):
            rows.append(dict(sweep_name=sn, strategy=strat, tau=tau, seed=seed,
                             episode=e["episode"], test_acc=e["test_acc"]))
    # keep only the latest sweep if several
    if sweep_names:
        latest = sorted(sweep_names)[-1]
        rows = [r for r in rows if r["sweep_name"] == latest]
    return rows, (sorted(sweep_names)[-1] if sweep_names else None)


def seed_avg(rows):
    """(strategy,tau,episode) -> mean test_acc."""
    acc = defaultdict(list)
    for r in rows:
        acc[(r["strategy"], r["tau"], r["episode"])].append(r["test_acc"])
    return {k: statistics.mean(v) for k, v in acc.items()}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    before = load_before()
    after, after_sweep = load_after()
    print(f"before rows: {len(before)}  after rows: {len(after)} (sweep={after_sweep})")

    # dump after csv
    if after:
        with open(OUTDIR / "after_curves.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["strategy", "tau", "seed", "episode", "test_acc"])
            w.writeheader()
            for r in sorted(after, key=lambda r: (r["strategy"], r["tau"], r["seed"], r["episode"])):
                w.writerow({k: r[k] for k in w.fieldnames})

    bavg = seed_avg(before)
    aavg = seed_avg(after) if after else {}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey="row")
    panels = [("before", "BEFORE  (buggy annotator + Adam ν′)"),
              ("after",  "AFTER  (seed-decoupled annotator + same Adam ν′)")]
    for ri, strat in enumerate(STRATS):
        for ci, (phase, title) in enumerate(panels):
            ax = axes[ri][ci]
            data = bavg if phase == "before" else aavg
            for tau in TAUS:
                xs = sorted({k[2] for k in data if k[0] == strat and k[1] == tau})
                ys = [data[(strat, tau, e)] for e in xs]
                if not xs:
                    continue
                ax.plot(xs, ys, marker="o", ms=3, lw=1.8,
                        color=TAU_COLOR[tau],
                        ls="-" if tau == 1.0 else "--",
                        label=TAU_LABEL[tau])
            ax.set_title(f"{strat}  —  {title}", fontsize=10)
            ax.grid(alpha=0.3)
            if ri == 1:
                ax.set_xlabel("AL episode")
            if ci == 0:
                ax.set_ylabel(f"{strat}\ntest acc (%)")
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("4_15 reproduction: oracle (τ=1) vs NoAA-noisy at ν′(τ) — before vs after the noise fix\n"
                 "CIFAR100 · linear-on-SimCLR · B=100 · 11 episodes · seed-avg(1,2,3)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUTDIR / "repro_4_15_before_after.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)

    # gap table at ep10
    if aavg:
        rows = []
        for strat in STRATS:
            orc_b = bavg.get((strat, 1.0, 10))
            orc_a = aavg.get((strat, 1.0, 10))
            for tau in [0.8, 0.5, 0.2]:
                nb = bavg.get((strat, tau, 10))
                na = aavg.get((strat, tau, 10))
                rows.append(dict(strategy=strat, tau=tau,
                                 oracle_before=round(orc_b, 2) if orc_b else None,
                                 noisy_before=round(nb, 2) if nb else None,
                                 gap_before=round(orc_b - nb, 2) if (orc_b and nb) else None,
                                 oracle_after=round(orc_a, 2) if orc_a else None,
                                 noisy_after=round(na, 2) if na else None,
                                 gap_after=round(orc_a - na, 2) if (orc_a and na) else None))
        with open(OUTDIR / "gap_table.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("\n=== ep10 seed-avg gap (oracle - noisy); negative => noisy BEATS oracle ===")
        print(f"{'strat':10s} {'tau':>4s} | {'orc_b':>6s} {'noi_b':>6s} {'gap_b':>6s} | "
              f"{'orc_a':>6s} {'noi_a':>6s} {'gap_a':>6s}")
        for r in rows:
            print(f"{r['strategy']:10s} {r['tau']:>4.1f} | {r['oracle_before']:>6} {r['noisy_before']:>6} "
                  f"{r['gap_before']:>6} | {r['oracle_after']:>6} {r['noisy_after']:>6} {r['gap_after']:>6}")


if __name__ == "__main__":
    main()
