"""
sweep_nu_prime.py
-----------------
Two modes:

  --submit   Submit one SLURM job per (method, budget, tau, seed).
             Each job runs estimating_delta_nu.py directly and saves a JSON
             to results/cifar100/nu_prime_raw/<method>_<budget>_tau<tau>_seed<seed>.json

  --average  Read all raw result files, print a per-seed table so you can
             inspect the variance, then average over seeds and write to
             results/cifar100/nu_prime_cache.json (used by train_al.py).

Workflow
--------
  # 1. dry-run: see the 243 jobs without submitting
  python batch_utils/sweep_nu_prime.py

  # 2. submit
  python batch_utils/sweep_nu_prime.py --submit

  # 3. after jobs finish: inspect + write cache
  python batch_utils/sweep_nu_prime.py --average

  # 4. with --dry_run: see table without touching the cache
  python batch_utils/sweep_nu_prime.py --average --dry_run

  # narrow to specific methods/budgets
  python batch_utils/sweep_nu_prime.py --methods probcover random --budgets 100 --submit
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
from pathlib import Path

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
REPO_DIR         = Path(__file__).resolve().parents[1]
VENV_ACTIVATE    = REPO_DIR / "venv" / "bin" / "activate"
ESTIMATOR_SCRIPT = "deep_al/pycls/noise/estimating_delta_nu.py"
LOG_DIR          = REPO_DIR / "logs" / "sweep"
GENERATED_SCRIPT = REPO_DIR / "batch_utils" / "_generated_sweep_nu_prime.sh"

# ── SLURM settings ─────────────────────────────────────────────────────────
SLURM_LOGIN_NODE = "phoenix-gw"
SBATCH_BIN       = "/vol/slurm/phoenix/bindir/bin/sbatch"
SLURM_CONF       = "/vol/slurm/phoenix/slurm.conf"


def _git_user_email() -> str:
    try:
        result = subprocess.run(
            ["git", "config", "user.email"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        user = os.environ.get("USER", "unknown")
        return f"{user}@mail.huji.ac.il"


SLURM_OPTS = {
    "mem":           "32g",
    "cpus-per-task": "4",
    "time":          "4:00:00",
    "gres":          "gg:g0:1,vmem:20g",
    "mail-user":     _git_user_email(),
    "mail-type":     "FAIL,TIME_LIMIT",
    "output":        str(LOG_DIR / "sweep_nu_prime_%A_%a.out"),
}

# ── sweep grid ──────────────────────────────────────────────────────────────
DEFAULT_DATASET  = "cifar100"
DEFAULT_METHODS  = ["probcover", "maxherding", "random"]
DEFAULT_BUDGETS  = [100, 200, 500]
DEFAULT_TAUS     = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_SEEDS    = [0, 1, 2]


def raw_file(dataset: str, method: str, budget: int,
             tau: float, seed: int) -> Path:
    """Path to single-run result JSON (one per method/budget/tau/seed)."""
    d = REPO_DIR / "results" / dataset / "nu_prime_raw"
    return d / f"{method}_{budget}_tau{tau:.1f}_seed{seed}.json"


# ── job builder ─────────────────────────────────────────────────────────────

def build_jobs(dataset: str, methods: list[str], budgets: list[int],
               taus: list[float], seeds: list[int]) -> list[str]:
    """One command per (method, budget, tau, seed)."""
    jobs = []
    for method, budget, tau, seed in itertools.product(methods, budgets, taus, seeds):
        output = raw_file(dataset, method, budget, tau, seed)
        cmd = (
            f"python3 {ESTIMATOR_SCRIPT}"
            f" --dataset {dataset}"
            f" --model linear"
            f" --budget {budget}"
            f" --tau {tau}"
            f" --epochs 100"
            f" --al_method {method}"
            f" --noise_mode random"
            f" --representation_model simclr"
            f" --seed {seed}"
            f" --output_file {output}"
        )
        jobs.append(cmd)
    return jobs


def build_script(jobs: list[str]) -> str:
    """Generate a SLURM array bash script."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    lines = ["#!/bin/bash"]
    for k, v in SLURM_OPTS.items():
        lines.append(f"#SBATCH --{k}={v}")
    lines.append(f"#SBATCH --array=0-{len(jobs)-1}")
    lines.append("")
    lines.append("JOBS=(")
    for job in jobs:
        lines.append(f'  "{job}"')
    lines.append(")")
    lines.append("")
    lines.append('CMD="${JOBS[$SLURM_ARRAY_TASK_ID]}"')
    lines.append('echo "Running: $CMD"')
    lines.append("")
    lines.append(f"cd {REPO_DIR}")
    lines.append(f"source {VENV_ACTIVATE}")
    lines.append("")
    lines.append("eval $CMD")
    return "\n".join(lines) + "\n"


# ── averaging ─────────────────────────────────────────────────────────────────

def run_average(dataset: str, methods: list[str], budgets: list[int],
                taus: list[float], seeds: list[int],
                dry_run: bool = False) -> None:
    """
    For each (method, budget, tau): read per-seed raw files, print table,
    average over seeds, write to nu_prime_cache.json.
    """
    import sys
    sys.path.insert(0, str(REPO_DIR))
    from deep_al.pycls.noise.nu_prime_cache import upsert

    any_written = False

    for method, budget in itertools.product(methods, budgets):
        print(f"\n{'='*64}")
        print(f"  {method}  budget={budget}")
        print(f"{'='*64}")

        # header
        seed_cols = "".join(f"  seed{s}" for s in seeds)
        print(f"  tau   {seed_cols}     avg")
        print("  " + "-" * (8 + 8 * len(seeds) + 8))

        for tau in taus:
            seed_vals: list[float] = []
            cells = ""
            for seed in seeds:
                f = raw_file(dataset, method, budget, tau, seed)
                if f.exists():
                    with open(f) as fp:
                        nu = json.load(fp).get("nu_prime")
                    seed_vals.append(nu)
                    cells += f"  {nu:.4f}"
                else:
                    cells += "    --  "

            if seed_vals:
                avg = float(np.mean(seed_vals))
                avg_str = f"  {avg:.4f}"
                if not dry_run:
                    upsert(dataset, method, budget, tau, avg)
                    any_written = True
            else:
                avg_str = "  ----"

            print(f"  {tau:.1f}  {cells}{avg_str}")

        # oracle anchor (tau=1.0 → nu_prime=0.0)
        if not dry_run:
            upsert(dataset, method, budget, 1.0, 0.0)
        print(f"  1.0    (oracle anchor -> 0.0000)")

    print()
    if dry_run:
        print("Dry-run: cache was NOT modified. Remove --dry_run to write.")
    elif any_written:
        print(f"Cache updated: results/{dataset}/nu_prime_cache.json")
    else:
        print("Nothing written — no result files found yet.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sweep nu_prime: submit SLURM jobs or average seed results")
    parser.add_argument("--dataset",  default=DEFAULT_DATASET,
                        choices=["cifar10", "cifar100", "imagenet50"])
    parser.add_argument("--methods",  nargs="+", default=DEFAULT_METHODS,
                        choices=["probcover", "maxherding", "random", "margin"])
    parser.add_argument("--budgets",  nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--taus",     nargs="+", type=float, default=DEFAULT_TAUS)
    parser.add_argument("--seeds",    nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--submit",   action="store_true",
                        help="Build script and submit to SLURM")
    parser.add_argument("--average",  action="store_true",
                        help="Read raw result files and write averaged cache")
    parser.add_argument("--dry_run",  action="store_true",
                        help="With --average: print table but don't write cache")
    args = parser.parse_args()

    if args.average:
        run_average(args.dataset, args.methods, args.budgets,
                    args.taus, args.seeds, dry_run=args.dry_run)
        return

    # ── submit / dry-run ───────────────────────────────────────────────────
    jobs = build_jobs(args.dataset, args.methods, args.budgets,
                      args.taus, args.seeds)
    script = build_script(jobs)

    n = len(args.methods)
    b = len(args.budgets)
    t = len(args.taus)
    s = len(args.seeds)
    print(f"\nSweep plan: {len(jobs)} jobs  "
          f"({n} methods × {b} budgets × {t} taus × {s} seeds)")
    print(f"  dataset : {args.dataset}")
    print(f"  methods : {args.methods}")
    print(f"  budgets : {args.budgets}")
    print(f"  taus    : {args.taus}")
    print(f"  seeds   : {args.seeds}")
    print(f"  output  : results/{args.dataset}/nu_prime_raw/<method>_<budget>_tau<tau>_seed<seed>.json")
    print()
    for i, job in enumerate(jobs):
        print(f"  [{i:03d}] {job}")
    print()

    GENERATED_SCRIPT.write_text(script)
    print(f"Script written to: {GENERATED_SCRIPT}")

    if not args.submit:
        print("\nDry-run complete. Add --submit to send to SLURM.")
        print("When done, run:  python batch_utils/sweep_nu_prime.py --average")
        return

    submit_cmd = f"SLURM_CONF={SLURM_CONF} {SBATCH_BIN} {GENERATED_SCRIPT}"
    ssh_cmd = ["ssh", SLURM_LOGIN_NODE, submit_cmd]
    print(f"\nSubmitting: {' '.join(ssh_cmd)}")
    result = subprocess.run(ssh_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Submitted: {result.stdout.strip()}")
        print("\nWhen all jobs finish, run:")
        print("  python batch_utils/sweep_nu_prime.py --average")
    else:
        print(f"sbatch error:\n{result.stderr}")


if __name__ == "__main__":
    main()
