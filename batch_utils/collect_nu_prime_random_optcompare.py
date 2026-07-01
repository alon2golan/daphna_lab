#!/usr/bin/env python
"""
Collect the RANDOM nu_prime optimizer-comparison results (SGD + Adam) into one
table the random fig5.5 optcompare plotter consumes.

Reads per-run JSONs from:
  results/cifar100/nu_prime_random_sgd/    (job 30911681, sgd)
  results/cifar100/nu_prime_random_adam/   (job 30917094, adam)
each of shape {al_method, budget, tau, seed, nu_prime, optimizer, oracle_acc}.

Writes (mirroring nu_prime_optcompare_ALL.* but for random):
  results/cifar100/nu_prime_optcompare_random_ALL.csv   — tidy rows
  results/cifar100/nu_prime_optcompare_random_ALL.json  — {optimizer:{budget:{tau: nu_prime_mean}}}

The JSON is what plot_fig55_daphna_noaa_optcompare_random.py reads to draw
NoAA-SGD and NoAA-Adam.

  python batch_utils/collect_nu_prime_random_optcompare.py
"""
from __future__ import annotations
import glob, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

RES_DIRS = [
    Path("/cs/labs/daphna/alongol/projects/daphna_lab/results/cifar100/nu_prime_random_sgd"),
    Path("/cs/labs/daphna/alongol/projects/daphna_lab/results/cifar100/nu_prime_random_adam"),
]
OUT_DIR = Path("/cs/labs/daphna/alongol/projects/daphna_lab/results/cifar100")


def main():
    rows = []
    for d in RES_DIRS:
        for f in sorted(glob.glob(str(d / "*.json"))):
            r = json.load(open(f))
            rows.append({
                "optimizer":  r.get("optimizer", "?"),
                "budget":     int(r["budget"]),
                "tau":        round(float(r["tau"]), 4),
                "seed":       int(r["seed"]),
                "nu_prime":   float(r["nu_prime"]),
                "oracle_acc": float(r.get("oracle_acc", np.nan)),
            })
    if not rows:
        print(f"No result JSONs found in {[str(d) for d in RES_DIRS]}")
        return

    tidy = pd.DataFrame(rows).sort_values(["optimizer", "budget", "tau", "seed"]).reset_index(drop=True)
    csv_path = OUT_DIR / "nu_prime_optcompare_random_ALL.csv"
    tidy.to_csv(csv_path, index=False)

    wide = tidy.groupby(["optimizer", "budget", "tau"]).agg(
        nu_mean=("nu_prime", "mean"), n=("nu_prime", "size")
    ).reset_index()

    nested: dict = defaultdict(lambda: defaultdict(dict))
    for _, r in wide.iterrows():
        nested[r["optimizer"]][str(int(r["budget"]))][f"{r['tau']:.4f}"] = round(float(r["nu_mean"]), 6)
    json_path = OUT_DIR / "nu_prime_optcompare_random_ALL.json"
    json.dump({k: dict(v) for k, v in nested.items()}, open(json_path, "w"), indent=1)

    print(f"✓ {csv_path}  ({len(tidy)} rows)")
    print(f"✓ {json_path}")
    print("\nseed-avg nu_prime per (optimizer, budget, tau):")
    print(f"{'opt':>5} {'B':>5} {'tau':>5} {'n':>2}  {'nu_mean':>8}")
    for _, r in wide.iterrows():
        print(f"{r['optimizer']:>5} {int(r['budget']):>5} {r['tau']:>5} {int(r['n']):>2}  {r['nu_mean']:>8.4f}")


if __name__ == "__main__":
    main()
