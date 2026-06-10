#!/usr/bin/env python
"""
One-off experiment: re-estimate nu_prime at B=3000 in daphna_lab, twice —
  (1) optimizer=sgd   (Tomer-style: SGD+Nesterov+WD, lr=0.025)
  (2) optimizer=adam  (daphna_lab default, lr=1e-3)
— on a CLEAN lsets parquet (built from the fig55 runs; verified no padding),
to test whether the B=3000 nu_prime floor was caused by the padded lset
(Problem 1) or by the estimator/proxy itself (Problem 2), and whether the
optimizer matters (as it did at B=100: SGD 0.695 vs Adam 0.613).

Grid: budget=3000 × tau{0.1,0.25,0.5} × seeds{1,2,3} × opt{sgd,adam} = 18 tasks.
SLURM array capped at %6 (daphna group GPU etiquette).

  python batch_utils/sweep_nu_prime_optcompare.py            # dry-run, writes script
  python batch_utils/sweep_nu_prime_optcompare.py --submit   # writes script (sbatch via ssh)
"""
from __future__ import annotations
import argparse, itertools
from pathlib import Path

REPO_DIR      = Path(__file__).resolve().parents[1]
VENV_ACTIVATE = REPO_DIR / "venv" / "bin" / "activate"
ESTIMATOR     = "deep_al/pycls/noise/estimating_delta_nu.py"
LOG_DIR       = REPO_DIR / "logs" / "sweep"
OUT_DIR       = REPO_DIR / "results" / "cifar100" / "nu_prime_B3000_optcompare"
GENERATED     = REPO_DIR / "batch_utils" / "_generated_sweep_nu_prime_optcompare.sh"

MAX_CONCURRENT = 6
BUDGETS  = [100, 300, 1000]   # B=3000 already estimated (job 30741466); this fills the rest
TAUS     = [0.1, 0.25, 0.5]
SEEDS    = [1, 2, 3]
OPTS     = ["sgd", "adam"]
EPOCHS   = 100

SLURM_OPTS = {
    "mem":           "32g",
    "cpus-per-task": "4",
    "time":          "4:00:00",
    "gres":          "gg:g0:1,vmem:20g",
    "output":        str(LOG_DIR / "nu_optcompare_%A_%a.out"),
}


def out_file(opt: str, budget: int, tau: float, seed: int) -> Path:
    return OUT_DIR / f"probcover_B{budget}_tau{tau}_seed{seed}_{opt}.json"


def build_jobs() -> list[str]:
    jobs = []
    for opt, budget, tau, seed in itertools.product(OPTS, BUDGETS, TAUS, SEEDS):
        jobs.append(
            f"python3 {ESTIMATOR}"
            f" --dataset cifar100 --model linear --representation_model simclr"
            f" --al_method probcover --noise_mode random --cluster_method kmeans"
            f" --budget {budget} --tau {tau} --seed {seed} --epochs {EPOCHS}"
            f" --optimizer {opt}"
            f" --output_file {out_file(opt, budget, tau, seed)}"
        )
    return jobs


def build_script(jobs: list[str]) -> str:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    L = ["#!/bin/bash"]
    for k, v in SLURM_OPTS.items():
        L.append(f"#SBATCH --{k}={v}")
    L.append(f"#SBATCH --array=0-{len(jobs)-1}%{MAX_CONCURRENT}")
    L += ["", "JOBS=("]
    L += [f'  "{j}"' for j in jobs]
    L += [")", "", 'CMD="${JOBS[$SLURM_ARRAY_TASK_ID]}"', 'echo "Running: $CMD"', ""]
    L += [f"cd {REPO_DIR}", f"source {VENV_ACTIVATE}", "", "eval $CMD"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true", help="Write the generated script (then sbatch via ssh phoenix-gw).")
    args = ap.parse_args()

    jobs = build_jobs()
    print(f"Sweep plan: {len(jobs)} jobs (budgets={BUDGETS} × {len(TAUS)} taus × {len(SEEDS)} seeds × {len(OPTS)} opts), cap %{MAX_CONCURRENT}")
    for i, j in enumerate(jobs):
        print(f"  [{i:02d}] {j}")
    GENERATED.write_text(build_script(jobs))
    print(f"\nScript written: {GENERATED}")
    if not args.submit:
        print("Dry-run. Add --submit to write the script for sbatch.")
    else:
        print("Submit with:")
        print(f'  ssh phoenix-gw \'bash -lc "SLURM_CONF=/vol/slurm/phoenix/slurm.conf '
              f'/vol/slurm/phoenix/bindir/bin/sbatch {GENERATED}"\'')


if __name__ == "__main__":
    main()
