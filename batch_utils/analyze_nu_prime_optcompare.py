#!/usr/bin/env python
"""
Analyze the B=3000 nu_prime optimizer-comparison experiment
(sweep_nu_prime_optcompare.py, job 30741076).

Reads results/cifar100/nu_prime_B3000_optcompare/*.json, averages nu_prime over
seeds per (optimizer, tau), and prints a comparison against:
  - the OLD floored estimator values (oracle-teacher-proj noaa_v_hat_wide.csv, B=3000)
  - the TRUE-LABEL crossover reference from fig55 (where noisy=oracle):
        tau=0.1 -> ~0.75,  tau=0.25 -> ~0.45,  tau=0.5 -> ~0.32

Interpretation:
  - If SGD (clean lsets) un-floors to ~the true crossover  -> the floor was padding+optimizer.
  - If both SGD and Adam still floor (~0) on clean lsets   -> Problem 2 (proxy saturation) confirmed.
  - SGD vs Adam gap quantifies the optimizer effect at B=3000 (cf. B=100: 0.695 vs 0.613).
"""
from __future__ import annotations
import glob, json
from collections import defaultdict
import numpy as np

RES = "/cs/labs/daphna/alongol/projects/daphna_lab/results/cifar100/nu_prime_B3000_optcompare"
TAUS = [0.1, 0.25, 0.5]
TRUE_CROSSOVER = {0.1: 0.75, 0.25: 0.45, 0.5: 0.32}   # from fig55 true-label noisy=oracle
OLD_FLOORED   = {0.1: 0.0039, 0.25: 0.0039, 0.5: 0.0117}  # oracle-teacher-proj csv (buggy)

def main():
    by = defaultdict(lambda: {"nu": [], "oacc": [], "seeds": []})
    for f in glob.glob(f"{RES}/*.json"):
        d = json.load(open(f))
        key = (d.get("optimizer", "?"), round(d["tau"], 4))
        by[key]["nu"].append(d["nu_prime"])
        by[key]["oacc"].append(d.get("oracle_acc", float("nan")))
        by[key]["seeds"].append(d["seed"])

    if not by:
        print(f"No results yet in {RES}")
        return

    print(f"{'opt':>5} {'tau':>5} {'n':>2}  {'nu_prime(mean±std)':>20}  {'oracle_acc':>11}  {'old(floored)':>12}  {'true xover':>10}")
    for opt in ["sgd", "adam"]:
        for tau in TAUS:
            k = (opt, round(tau, 4))
            if k not in by:
                print(f"{opt:>5} {tau:>5}  -- (pending)")
                continue
            nu = np.array(by[k]["nu"]); oa = np.array(by[k]["oacc"])
            print(f"{opt:>5} {tau:>5} {len(nu):>2}  {nu.mean():>10.4f} ± {nu.std():>6.4f}  "
                  f"{np.nanmean(oa):>11.4f}  {OLD_FLOORED[tau]:>12.4f}  {TRUE_CROSSOVER[tau]:>10.2f}")
        print()

if __name__ == "__main__":
    main()
