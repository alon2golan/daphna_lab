#!/usr/bin/env python
"""
Cache sweep: re-estimate nu_prime over a dense budget grid for probcover AND
random, optimizer=adam, to populate the nu_prime curve cache.

Grid: al_method{probcover,random} x budget{100,200,...,1000} x tau{0.1,0.25,0.5}
      x seed{1..5} x opt{adam}  = 2*10*3*5 = 300 tasks.
SLURM array capped at %6 (daphna group GPU etiquette).

Data sufficiency (verified 2026-06-10): estimator uses noisy_size=int(budget/tau)
and reads the resnet18 probcover parquet. tau=0.1/B=1000 -> noisy_size=10000,
which exists exactly (oracle B=1000@0.1 and noisy B=10000@0.1 both present).
random bypasses the parquet (np.random.choice over the >=30k train pool).

  python batch_utils/sweep_nu_prime_cache_pcrand.py            # dry-run, writes script
  python batch_utils/sweep_nu_prime_cache_pcrand.py --submit   # writes script (sbatch via ssh)
"""
from __future__ import annotations
import argparse, itertools
from pathlib import Path

REPO_DIR      = Path(__file__).resolve().parents[1]
VENV_ACTIVATE = REPO_DIR / "venv" / "bin" / "activate"
ESTIMATOR     = "deep_al/pycls/noise/estimating_delta_nu.py"
LOG_DIR       = REPO_DIR / "logs" / "sweep"
OUT_DIR       = REPO_DIR / "results" / "cifar100" / "nu_prime_cache_pcrand"
GENERATED     = REPO_DIR / "batch_utils" / "_generated_sweep_nu_prime_cache_pcrand.sh"

MAX_CONCURRENT = 8
AL_METHODS = ["probcover", "random"]
BUDGETS    = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
TAUS       = [0.1, 0.25, 0.5]
SEEDS      = [1, 2, 3, 4, 5]
OPT        = "adam"
EPOCHS     = 100

SLURM_OPTS = {
    "mem":           "32g",
    "cpus-per-task": "4",
    "time":          "4:00:00",
    "gres":          "gg:g0:1,vmem:20g",
    "output":        str(LOG_DIR / "nu_cache_pcrand_%A_%a.out"),
}


def out_file(al: str, budget: int, tau: float, seed: int) -> Path:
    return OUT_DIR / f"{al}_B{budget}_tau{tau}_seed{seed}_{OPT}.json"


def build_jobs() -> list[str]:
    jobs = []
    for al, budget, tau, seed in itertools.product(AL_METHODS, BUDGETS, TAUS, SEEDS):
        jobs.append(
            f"python3 {ESTIMATOR}"
            f" --dataset cifar100 --model linear --representation_model simclr"
            f" --al_method {al} --noise_mode random --cluster_method kmeans"
            f" --budget {budget} --tau {tau} --seed {seed} --epochs {EPOCHS}"
            f" --optimizer {OPT}"
            f" --output_file {out_file(al, budget, tau, seed)}"
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
    print(f"Sweep plan: {len(jobs)} jobs "
          f"(al={AL_METHODS} x budgets={BUDGETS} x {len(TAUS)} taus x {len(SEEDS)} seeds x opt={OPT}), "
          f"cap %{MAX_CONCURRENT}")
    for i, j in enumerate(jobs):
        print(f"  [{i:03d}] {j}")
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
