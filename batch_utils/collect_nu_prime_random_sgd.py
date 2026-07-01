#!/usr/bin/env python
"""
Collect the RANDOM x SGD nu_prime sweep (job 30911681) into a table.

Reads every result JSON in results/cifar100/nu_prime_random_sgd/ produced by
sweep_nu_prime_random_sgd.py (al=random, opt=sgd, B{100,300,1000,3000} x
tau{0.1,0.25,0.5} x seed{1,2,3} = 36 runs), and writes:

  results/cifar100/nu_prime_random_sgd_ALL.csv   — tidy: al_method,optimizer,budget,tau,seed,nu_prime,oracle_acc
  results/cifar100/nu_prime_random_sgd_ALL.json  — nested {budget:{tau: mean nu_prime}}

This fills the random x SGD column the probcover-only optcompare
(nu_prime_optcompare_ALL.*) left empty. Kept in a SEPARATE dir/file ON PURPOSE
so it never silently blends into the all-probcover optcompare table.
"""
from __future__ import annotations
import glob, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

RES_DIR = Path("/cs/labs/daphna/alongol/projects/daphna_lab/results/cifar100/nu_prime_random_sgd")
OUT_DIR = Path("/cs/labs/daphna/alongol/projects/daphna_lab/results/cifar100")

# For side-by-side context (from memory: probcover SGD, random Adam@cache), tau=0.1 unless noted.
REF = {
    "probcover_sgd": {100: 0.704, 300: 0.757, 1000: 0.423, 3000: 0.113},  # tau=0.1
    "random_adam":   {100: 0.723, 300: 0.646, 1000: 0.493},               # tau=0.1 (no B=3000)
}

def main():
    rows = []
    for f in sorted(glob.glob(str(RES_DIR / "*.json"))):
        d = json.load(open(f))
        rows.append({
            "al_method": d.get("al_method", "random"),
            "optimizer": d.get("optimizer", "sgd"),
            "budget":    int(d["budget"]),
            "tau":       round(float(d["tau"]), 4),
            "seed":      int(d["seed"]),
            "nu_prime":  float(d["nu_prime"]),
            "oracle_acc": float(d.get("oracle_acc", np.nan)),
        })
    if not rows:
        print(f"No result JSONs in {RES_DIR}")
        return
    tidy = pd.DataFrame(rows).sort_values(["budget", "tau", "seed"]).reset_index(drop=True)
    csv_path = OUT_DIR / "nu_prime_random_sgd_ALL.csv"
    tidy.to_csv(csv_path, index=False)

    wide = tidy.groupby(["budget", "tau"]).agg(
        nu_mean=("nu_prime", "mean"), nu_std=("nu_prime", "std"),
        n=("nu_prime", "size"), oacc=("oracle_acc", "mean")).reset_index()

    # nested {budget:{tau: mean nu_prime}}
    nested: dict = defaultdict(dict)
    for _, r in wide.iterrows():
        nested[str(int(r["budget"]))][f"{r['tau']:.4f}"] = round(float(r["nu_mean"]), 6)
    json_path = OUT_DIR / "nu_prime_random_sgd_ALL.json"
    json.dump(dict(nested), open(json_path, "w"), indent=2)

    print(f"collected {len(tidy)} runs -> {csv_path}\n              -> {json_path}\n")
    print(f"{'B':>5} {'tau':>5} {'n':>2}  {'nu_prime(mean±std)':>20}  {'oracle_acc':>10}")
    for _, r in wide.iterrows():
        std = r["nu_std"] if not np.isnan(r["nu_std"]) else 0.0
        print(f"{int(r['budget']):>5} {r['tau']:>5} {int(r['n']):>2}  "
              f"{r['nu_mean']:>10.4f} ± {std:>6.4f}  {r['oacc']:>10.4f}")

    # tau=0.1 cross-method context
    print("\nτ=0.1 ν' across methods (this sweep = random/SGD):")
    print(f"{'B':>5} {'random/SGD':>11} {'probcover/SGD':>14} {'random/Adam':>12}")
    t01 = wide[wide["tau"] == 0.1].set_index("budget")["nu_mean"].to_dict()
    for B in [100, 300, 1000, 3000]:
        rs = f"{t01[B]:.3f}" if B in t01 else "—"
        ps = f"{REF['probcover_sgd'].get(B, '—')}"
        ra = f"{REF['random_adam'].get(B, '—')}"
        print(f"{B:>5} {rs:>11} {ps:>14} {ra:>12}")

if __name__ == "__main__":
    main()
