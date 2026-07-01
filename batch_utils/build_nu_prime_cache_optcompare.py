#!/usr/bin/env python
"""
Build ONE consolidated nu_prime cache for the 4-arm optimizer comparison:
{random, probcover} x {adam, sgd}, over B{100,300,1000,3000} x tau{0.1,0.25,0.5}.

Pulls each arm from its already-collected source (no re-running):
  probcover/{adam,sgd} <- nu_prime_optcompare_ALL.json         (jobs 30741466/30741961)
  random/{adam,sgd}    <- nu_prime_optcompare_random_ALL.json  (jobs 30917094 adam / 30911681 sgd)

NOTE: random/adam is the per-(B,τ) NoAA-threshold estimate (job 30917094, the one
the fig55_..._random.png plot actually uses) — NOT nu_prime_cache.json["random"],
which is the dense per-ν cache at budgets {100,200,500} and has no B=3000.

Writes:
  results/cifar100/nu_prime_cache_optcompare.json  — nested {al:{opt:{budget:{tau: nu'}}}}
  results/cifar100/nu_prime_cache_optcompare.csv   — tidy: al_method,optimizer,budget,tau,nu_prime,source

All four arms cover the full B{100,300,1000,3000} x tau{0.1,0.25,0.5} grid.
This file is SEPARATE from the live nu_prime_cache.json (which has no optimizer
axis and is implicitly all-adam) — that one is NOT touched.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

RES = Path("/cs/labs/daphna/alongol/projects/daphna_lab/results/cifar100")
BUDGETS = ["100", "300", "1000", "3000"]
TAUS = ["0.1000", "0.2500", "0.5000"]

def load(name):
    return json.load(open(RES / name))

def main():
    pc  = load("nu_prime_optcompare_ALL.json")         # {opt:{B:{tau}}} probcover
    rnd = load("nu_prime_optcompare_random_ALL.json")  # {opt:{B:{tau}}} random

    # (al, opt) -> ({B:{tau}} source dict, source label)
    arms = {
        ("probcover", "adam"): (pc["adam"],  "nu_prime_optcompare_ALL.json[adam] (job 30741961/30741466)"),
        ("probcover", "sgd"):  (pc["sgd"],   "nu_prime_optcompare_ALL.json[sgd] (job 30741961/30741466)"),
        ("random",    "adam"): (rnd["adam"], "nu_prime_optcompare_random_ALL.json[adam] (job 30917094)"),
        ("random",    "sgd"):  (rnd["sgd"],  "nu_prime_optcompare_random_ALL.json[sgd] (job 30911681)"),
    }

    nested: dict = {}
    rows = []
    gaps = []
    for (al, opt), (src, label) in arms.items():
        nested.setdefault(al, {})[opt] = {}
        for B in BUDGETS:
            cell = {}
            for tau in TAUS:
                v = src.get(B, {}).get(tau)
                if v is None:
                    gaps.append(f"{al}/{opt} B={B} tau={tau}")
                    continue
                v = round(float(v), 6)
                cell[tau] = v
                rows.append({"al_method": al, "optimizer": opt, "budget": int(B),
                             "tau": float(tau), "nu_prime": v, "source": label})
            if cell:
                nested[al][opt][B] = cell

    json_path = RES / "nu_prime_cache_optcompare.json"
    json.dump(nested, open(json_path, "w"), indent=2)
    tidy = pd.DataFrame(rows).sort_values(["al_method", "optimizer", "budget", "tau"]).reset_index(drop=True)
    csv_path = RES / "nu_prime_cache_optcompare.csv"
    tidy.to_csv(csv_path, index=False)

    print(f"wrote {len(tidy)} cells -> {json_path}\n              -> {csv_path}\n")
    print("ν' by arm (rows=B, cols=τ 0.1/0.25/0.5):")
    for al in ["probcover", "random"]:
        for opt in ["adam", "sgd"]:
            print(f"  {al}/{opt}:")
            for B in BUDGETS:
                cell = nested[al][opt].get(B)
                if cell:
                    print(f"    B={B:>4}: " + "  ".join(f"{cell.get(t, float('nan')):.3f}" for t in TAUS))
                else:
                    print(f"    B={B:>4}: (none)")
    if gaps:
        print("\nGAPS (not cached, never estimated):")
        for g in gaps:
            print("  - " + g)

if __name__ == "__main__":
    main()
