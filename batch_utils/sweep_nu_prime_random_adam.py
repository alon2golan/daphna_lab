#!/usr/bin/env python
"""
Fill the missing RANDOM × ADAM nu_prime column at the fig55 grid.

Twin of sweep_nu_prime_random_sgd.py (job 30911681, which produced the SGD
column in results/cifar100/nu_prime_random_sgd/). This one re-estimates nu' for
RANDOM with the ADAM optimizer on the SAME estimator/grid, so the random fig5.5
optcompare plot has both NoAA-SGD and NoAA-Adam — matching the probcover
fig55_daphna_noaa_optcompare figure.

The existing random Adam values (nu_prime_cache.json) are the DENSE per-nu
crossover cache at different budgets {100,200,500,...} — wrong structure/grid
for the per-(B,tau) NoAA threshold the optcompare plotter needs. Hence a fresh
Adam sweep on the {100,300,1000,3000} × {0.1,0.25,0.5} grid.

For random, the estimator's get_train_idx ignores the parquet entirely
(np.random.choice(train_idx, size=budget)) -> NO parquet/padding/snapping issues.

Grid: al=random × opt=adam × budget{100,300,1000,3000} × tau{0.1,0.25,0.5}
      × seed{1,2,3} = 36 tasks. SLURM array capped at %6 (group GPU etiquette).

  python batch_utils/sweep_nu_prime_random_adam.py            # dry-run, writes script
  python batch_utils/sweep_nu_prime_random_adam.py --submit   # writes script (sbatch via ssh)
"""
from __future__ import annotations
import argparse, itertools, subprocess
from pathlib import Path

REPO_DIR      = Path(__file__).resolve().parents[1]
VENV_ACTIVATE = REPO_DIR / "venv" / "bin" / "activate"
ESTIMATOR     = "deep_al/pycls/noise/estimating_delta_nu.py"
LOG_DIR       = REPO_DIR / "logs" / "sweep"
OUT_DIR       = REPO_DIR / "results" / "cifar100" / "nu_prime_random_adam"
GENERATED     = REPO_DIR / "batch_utils" / "_generated_sweep_nu_prime_random_adam.sh"

MAX_CONCURRENT = 6
AL_METHOD = "random"
OPTS      = ["adam"]
BUDGETS   = [100, 300, 1000, 3000]
TAUS      = [0.1, 0.25, 0.5]
SEEDS     = [1, 2, 3]
EPOCHS    = 100

SLURM_LOGIN_NODE = "phoenix-gw"
SBATCH_BIN       = "/vol/slurm/phoenix/bindir/bin/sbatch"
SLURM_CONF       = "/vol/slurm/phoenix/slurm.conf"

SLURM_OPTS = {
    "mem":           "32g",
    "cpus-per-task": "4",
    "time":          "4:00:00",
    "gres":          "gg:g0:1,vmem:20g",
    "output":        str(LOG_DIR / "nu_random_adam_%A_%a.out"),
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
    ap.add_argument("--submit", action="store_true", help="Write script and sbatch via ssh phoenix.")
    args = ap.parse_args()

    jobs = build_jobs()
    print(f"Sweep plan: {len(jobs)} jobs (al={AL_METHOD} × {OPTS} × budgets={BUDGETS} × {len(TAUS)} taus × {len(SEEDS)} seeds), cap %{MAX_CONCURRENT}")
    for i, j in enumerate(jobs):
        print(f"  [{i:02d}] {j}")
    GENERATED.write_text(build_script(jobs))
    print(f"\nScript written: {GENERATED}")
    if not args.submit:
        print("Dry-run. Add --submit to sbatch via ssh phoenix.")
        return

    submit_cmd = f"SLURM_CONF={SLURM_CONF} {SBATCH_BIN} {GENERATED}"
    r = subprocess.run(["ssh", SLURM_LOGIN_NODE, submit_cmd],
                       capture_output=True, text=True)
    print("rc:", r.returncode, "| out:", r.stdout.strip(), "| err:", r.stderr.strip())


if __name__ == "__main__":
    main()
