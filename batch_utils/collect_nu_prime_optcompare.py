#!/usr/bin/env python
"""
Collect ALL daphna_lab nu_prime optimizer-comparison results into ONE place.

Reads every result JSON in results/cifar100/nu_prime_B3000_optcompare/ (despite the
dir name it now holds B=100/300/1000/3000), produced by sweep_nu_prime_optcompare.py
(jobs 30741466 for B=3000, 30741961 for B=100/300/1000), and writes:

  results/cifar100/nu_prime_optcompare_ALL.csv   — tidy: optimizer,budget,tau,seed,nu_prime,oracle_acc
  results/cifar100/nu_prime_optcompare_ALL.json  — nested {optimizer:{budget:{tau: nu_prime_mean}}}

The JSON is what the fig5.5 plotter consumes to draw NoAA-SGD and NoAA-Adam.
"""
from __future__ import annotations
import glob, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

RES_DIR = Path("/cs/labs/daphna/alongol/projects/daphna_lab/results/cifar100/nu_prime_B3000_optcompare")
OUT_DIR = Path("/cs/labs/daphna/alongol/projects/daphna_lab/results/cifar100")
TRUE_CROSSOVER = {  # fig55 true-label noisy=oracle reference, per budget (B=3000 measured; others ~)
    3000: {0.1: 0.75, 0.25: 0.45, 0.5: 0.32},
}

def main():
    rows = []
    for f in sorted(glob.glob(str(RES_DIR / "*.json"))):
        d = json.load(open(f))
        rows.append({
            "optimizer": d.get("optimizer", "?"),
            "budget":    int(d["budget"]),
            "tau":       round(float(d["tau"]), 4),
            "seed":      int(d["seed"]),
            "nu_prime":  float(d["nu_prime"]),
            "oracle_acc": float(d.get("oracle_acc", np.nan)),
        })
    if not rows:
        print(f"No result JSONs in {RES_DIR}")
        return
    tidy = pd.DataFrame(rows).sort_values(["optimizer", "budget", "tau", "seed"]).reset_index(drop=True)
    csv_path = OUT_DIR / "nu_prime_optcompare_ALL.csv"
    tidy.to_csv(csv_path, index=False)

    # nested {opt:{budget:{tau: mean nu_prime}}}
    nested: dict = defaultdict(lambda: defaultdict(dict))
    wide = tidy.groupby(["optimizer", "budget", "tau"]).agg(
        nu_mean=("nu_prime", "mean"), nu_std=("nu_prime", "std"),
        n=("nu_prime", "size"), oacc=("oracle_acc", "mean")).reset_index()
    for _, r in wide.iterrows():
        nested[r["optimizer"]][str(int(r["budget"]))][f"{r['tau']:.4f}"] = round(float(r["nu_mean"]), 6)
    json_path = OUT_DIR / "nu_prime_optcompare_ALL.json"
    json.dump({k: dict(v) for k, v in nested.items()}, open(json_path, "w"), indent=2)

    # console summary
    print(f"collected {len(tidy)} runs -> {csv_path}\n              -> {json_path}\n")
    print(f"{'opt':>5} {'B':>5} {'tau':>5} {'n':>2}  {'nu_prime(mean±std)':>20}  {'oracle_acc':>10}")
    for _, r in wide.iterrows():
        xo = TRUE_CROSSOVER.get(int(r["budget"]), {}).get(r["tau"])
        xs = f"  (true≈{xo})" if xo is not None else ""
        print(f"{r['optimizer']:>5} {int(r['budget']):>5} {r['tau']:>5} {int(r['n']):>2}  "
              f"{r['nu_mean']:>10.4f} ± {(r['nu_std'] if not np.isnan(r['nu_std']) else 0):>6.4f}  {r['oacc']:>10.4f}{xs}")

if __name__ == "__main__":
    main()
