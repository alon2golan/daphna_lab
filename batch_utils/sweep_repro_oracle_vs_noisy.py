"""
sweep_repro_oracle_vs_noisy.py
------------------------------
Reproduce the 2026_4_15 oracle-vs-noisy AL comparison AFTER the noise
seed-decoupling fix (commit ade1fd2).

Original observation (output/CIFAR100/resnet18/2026_4_15/, probcover, B=100,
11 episodes, seed 1):
    oracle ep10 = 47.74
    tau=0.8     = 46.70   (gap 1.04)
    tau=0.5     = 44.52   (gap 3.22)   <- nu' from cache = 0.259
    tau=0.2     = 42.64   (gap 5.10)
Per Tomer the noisy run at the equilibrium nu'(tau) should ≈ oracle; pre-fix
the oracle won and the gap grew across episodes. The bug: the annotator's
noise RNG was coupled to cfg.RNG_SEED, inflating realized lset noise far above
nominal. This sweep re-runs the same grid so we can re-measure the gap.

Each noisy job passes --tau τ; train_al.configure_nu_prime_annotator looks up
nu'(probcover, B=100, τ) from results/cifar100/nu_prime_cache.json (the value
is unchanged by the fix — the estimator generates its own noise and never used
the buggy annotator). Oracle = --target_nu 0.0 (tau→1, nu'→0), matching 4_15.

Usage:
    python batch_utils/sweep_repro_oracle_vs_noisy.py            # dry-run
    python batch_utils/sweep_repro_oracle_vs_noisy.py --submit   # sbatch it
"""
from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime
from pathlib import Path

# Import sweep_train_al.py without needing batch_utils to be a package.
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "sweep_train_al", _HERE / "sweep_train_al.py"
)
swt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(swt)

CFG = "deep_al/configs/cifar100/al/RESNET18.yaml"

# Mirror the 4_15 probcover grid: oracle + three taus.
AL_CONFIGS = [
    f"--cfg {CFG} --target_nu 0.0",   # oracle (tau=1, nu'=0)
    f"--cfg {CFG} --tau 0.8",         # nu' looked up from cache
    f"--cfg {CFG} --tau 0.5",
    f"--cfg {CFG} --tau 0.2",
]
GRID = dict(al=["probcover"], seed=[1, 2, 3])
FIXED_ARGS = (
    "--initial_delta 0.75 "
    "--budget 100 "
    "--eval_model_type from_features "
    "--max_iter 10 "
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true",
                    help="Write the array script and sbatch it (via phoenix-gw).")
    args = ap.parse_args()

    sweep_name = f"repro4_15_oracle_vs_noisy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    jobs = swt.build_jobs_from(AL_CONFIGS, GRID, FIXED_ARGS, sweep_name=sweep_name)

    print(f"\nsweep_name: {sweep_name}")
    print(f"{len(jobs)} jobs ({len(AL_CONFIGS)} configs x "
          f"{len(GRID['al'])} al x {len(GRID['seed'])} seeds):\n")
    for i, j in enumerate(jobs):
        print(f"  [{i:02d}] {j}")

    if not args.submit:
        print("\nDry-run. Add --submit to send to SLURM.")
        return

    res = swt.submit_sbatch_jobs(jobs)
    if res["returncode"] == 0:
        print(f"\nSubmitted: {res['stdout']}")
    else:
        print(f"\nsbatch failed (rc={res['returncode']}):\n{res['stderr']}")


if __name__ == "__main__":
    main()
