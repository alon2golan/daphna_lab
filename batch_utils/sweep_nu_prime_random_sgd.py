#!/usr/bin/env python
"""
Fill the missing RANDOM × SGD nu_prime column.

The earlier optimizer-comparison sweep (sweep_nu_prime_optcompare.py ->
nu_prime_optcompare_ALL.csv) was probcover-ONLY, so we have SGD-vs-Adam nu'
for probcover but only Adam (from the dense cache) for random. This sweep
re-estimates nu' for RANDOM with the Tomer-style SGD optimizer, on the SAME
estimator/grid, so random gets a comparable SGD column.

For random, the estimator's get_train_idx ignores the parquet entirely
(np.random.choice(train_idx, size=budget)), so there are NO parquet/padding/
snapping issues -- including at B=3000 (noisy_size=B/tau=30000 < 50000 pool).

Grid: al=random × opt=sgd × budget{100,300,1000,3000} × tau{0.1,0.25,0.5}
      × seed{1,2,3} = 36 tasks. SLURM array capped at %6 (group GPU etiquette).

Results land in a SEPARATE dir (results/cifar100/nu_prime_random_sgd/) so the
existing collect_nu_prime_optcompare.py (which does NOT read al_method) cannot
blend them into the all-probcover nu_prime_optcompare_ALL.csv. Each JSON still
carries al_method/optimizer/oracle_acc for downstream integration.

  python batch_utils/sweep_nu_prime_random_sgd.py            # dry-run, writes script
  python batch_utils/sweep_nu_prime_random_sgd.py --submit   # writes script (sbatch via ssh)
"""
from __future__ import annotations
import argparse, itertools
from pathlib import Path

REPO_DIR      = Path(__file__).resolve().parents[1]
VENV_ACTIVATE = REPO_DIR / "venv" / "bin" / "activate"
ESTIMATOR     = "deep_al/pycls/noise/estimating_delta_nu.py"
LOG_DIR       = REPO_DIR / "logs" / "sweep"
OUT_DIR       = REPO_DIR / "results" / "cifar100" / "nu_prime_random_sgd"
GENERATED     = REPO_DIR / "batch_utils" / "_generated_sweep_nu_prime_random_sgd.sh"

MAX_CONCURRENT = 6
AL_METHOD = "random"
OPTS      = ["sgd"]
BUDGETS   = [100, 300, 1000, 3000]
TAUS      = [0.1, 0.25, 0.5]
SEEDS     = [1, 2, 3]
EPOCHS    = 100

SLURM_OPTS = {
    "mem":           "32g",
    "cpus-per-task": "4",
    "time":          "4:00:00",
    "gres":          "gg:g0:1,vmem:20g",
    "output":        str(LOG_DIR / "nu_random_sgd_%A_%a.out"),
}


def out_file(opt: str, budget: int, tau: float, seed: int) -> Path:
    return OUT_DIR / f"{AL_METHOD}_B{budget}_tau{tau}_seed{seed}_{opt}.json"


def build_jobs() -> list[str]:
    jobs = []
    for opt, budget, tau, seed in itertools.product(OPTS, BUDGETS, TAUS, SEEDS):
        jobs.append(
            f"python3 {ESTIMATOR}"
            f" --dataset cifar100 --model linear --representation_model simclr"
            f" --al_method {AL_METHOD} --noise_mode random --cluster_method kmeans"
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
    print(f"Sweep plan: {len(jobs)} jobs (al={AL_METHOD} × {OPTS} × budgets={BUDGETS} × {len(TAUS)} taus × {len(SEEDS)} seeds), cap %{MAX_CONCURRENT}")
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
