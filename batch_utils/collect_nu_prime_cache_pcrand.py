#!/usr/bin/env python
"""
Collect the dense ν' cache sweep (job 30797616, probcover+random, adam, 5 seeds)
into the dashboard's curve cache.

Reads every per-task JSON in
  results/cifar100/nu_prime_cache_pcrand/{al}_B{b}_tau{t}_seed{s}_adam.json
(grid: al_method{probcover,random} x budget{100..1000 step100} x tau{0.1,0.25,0.5} x seed{1..5}),
means over the 5 seeds keyed strategy->budget->tau, and writes:

  results/cifar100/nu_prime_cache_pcrand.csv        — tidy per-cell rows
  results/cifar100/nu_prime_cache_pcrand.json       — nested {strategy:{budget:{tau: nu_mean}}}

Then MERGES the nested means into the live dashboard cache
  results/cifar100/nu_prime_cache.json   (backed up to .bak_pcrand first)
using deep-merge: new (strategy,budget,tau) cells are added; overlapping cells are
overwritten with the new adam mean. A diff of every overwritten cell is printed.
"""
from __future__ import annotations
import glob, json, shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

RES_DIR = Path("/cs/labs/daphna/alongol/projects/daphna_lab/results/cifar100/nu_prime_cache_pcrand")
OUT_DIR = Path("/cs/labs/daphna/alongol/projects/daphna_lab/results/cifar100")
CACHE   = OUT_DIR / "nu_prime_cache.json"

def main():
    rows = []
    for f in sorted(glob.glob(str(RES_DIR / "*.json"))):
        d = json.load(open(f))
        rows.append({
            "strategy":   d["al_method"],
            "budget":     int(d["budget"]),
            "tau":        round(float(d["tau"]), 4),
            "seed":       int(d["seed"]),
            "nu_prime":   float(d["nu_prime"]),
            "oracle_acc": float(d.get("oracle_acc", np.nan)),
        })
    if not rows:
        print(f"No result JSONs in {RES_DIR}")
        return
    tidy = pd.DataFrame(rows).sort_values(["strategy", "budget", "tau", "seed"]).reset_index(drop=True)
    tidy.to_csv(OUT_DIR / "nu_prime_cache_pcrand.csv", index=False)

    wide = tidy.groupby(["strategy", "budget", "tau"]).agg(
        nu_mean=("nu_prime", "mean"), nu_std=("nu_prime", "std"),
        n=("nu_prime", "size"), oacc=("oracle_acc", "mean")).reset_index()

    nested: dict = defaultdict(lambda: defaultdict(dict))
    for _, r in wide.iterrows():
        nested[r["strategy"]][str(int(r["budget"]))][f"{r['tau']:.4f}"] = round(float(r["nu_mean"]), 6)
    nested = {k: dict(v) for k, v in nested.items()}
    json.dump(nested, open(OUT_DIR / "nu_prime_cache_pcrand.json", "w"), indent=2)

    # console summary of the new means
    print(f"collected {len(tidy)} runs (2 methods x 10 budgets x 3 tau x 5 seeds)")
    print(f"{'strat':>9} {'B':>5} {'tau':>6} {'n':>2}  {'nu_prime(mean±std)':>20}  {'oracle_acc':>10}")
    for _, r in wide.iterrows():
        print(f"{r['strategy']:>9} {int(r['budget']):>5} {r['tau']:>6} {int(r['n']):>2}  "
              f"{r['nu_mean']:>10.4f} ± {(r['nu_std'] if not np.isnan(r['nu_std']) else 0):>6.4f}  {r['oacc']:>10.4f}")

    # ---- merge into the live cache ----
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    shutil.copy(CACHE, CACHE.with_suffix(".json.bak_pcrand")) if CACHE.exists() else None
    overwrites, adds = [], 0
    for strat, budgets in nested.items():
        cache.setdefault(strat, {})
        for b, taus in budgets.items():
            cache[strat].setdefault(b, {})
            for t, v in taus.items():
                if t in cache[strat][b]:
                    overwrites.append((strat, b, t, cache[strat][b][t], v))
                else:
                    adds += 1
                cache[strat][b][t] = v
    json.dump(cache, open(CACHE, "w"), indent=2)

    print(f"\nmerged into {CACHE.name}: +{adds} new cells, {len(overwrites)} overwritten")
    if overwrites:
        print(f"{'strat':>9} {'B':>5} {'tau':>7}  {'old':>9} -> {'new':>9}   (delta)")
        for strat, b, t, old, new in overwrites:
            print(f"{strat:>9} {b:>5} {t:>7}  {old:>9.4f} -> {new:>9.4f}   ({new-old:+.4f})")

if __name__ == "__main__":
    main()
