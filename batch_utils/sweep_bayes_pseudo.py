"""
Launch the Bayesian-coverage §7.5 curriculum pseudo-labeling sweep
(docs/bayesian_coverage_NEXT_STEPS.md §7.5 + RESULTS §4.6).

All arms: al=bayes_product + TRAIN_LABEL_MODE=pseudo (labeled points train on
their soft posterior, exactly like soft_posterior; the arm adds soft-posterior
targets for confident UNLABELED points). Three pseudo arms:

  * all    — every covered (non-prior-only) pool point is pseudo-labeled.
  * lin10  — top-k most-confident per predicted class, k = training round
             (1 per class at 100 labels ... 10 per class at 1000).
  * agg50  — same, but k grows 1 -> 50 over the 10 rounds (steeper curriculum).

Baselines: the observed / soft_posterior arms of the trainmode sweep
(job 30980646, same regime + seeds) — NOT re-run here.

Grid: 3 arms x noise {0.0, 0.2, 0.5, 0.8} x seed {1,2,3} = 36 runs.
Same regime as the BAYESCOV sweeps: CIFAR100, budget 100 x 10 episodes,
tophat delta=0.75, from_features. Seeds {1,2,3} match the existing bayes
sweeps so per-seed comparison is paired.

Each job tags `noise=<level>` (dashboard facet via the noise alias) and
`pseudo=<arm>`; pseudo_scope / pseudo_k are also recorded as result.json
params (result_writer), so the three arms get distinct config_keys.

Usage:
    python batch_utils/sweep_bayes_pseudo.py             # dry-run: print plan
    python batch_utils/sweep_bayes_pseudo.py --smoke     # 3-job, 1-episode smoke plan
    python batch_utils/sweep_bayes_pseudo.py [--smoke] --submit
"""
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

from sweep_train_al import (
    build_jobs_from,
    SLURM_OPTS,
    SLURM_LOGIN_NODE,
    SBATCH_BIN,
    SLURM_CONF,
    REPO_DIR,
    VENV_ACTIVATE,
    LOG_DIR,
)
from sweep_bayes_coverage import build_script

CFG_DIR = "deep_al/configs/cifar100/al"
NOISE_CFGS = [
    ("BAYESCOV_n00.yaml", "0.0"),
    ("BAYESCOV_n20.yaml", "0.2"),
    ("BAYESCOV_n50.yaml", "0.5"),
    ("BAYESCOV_n80.yaml", "0.8"),
]
# arm name -> pseudo CLI args (schedule indexed by training round, 1-based)
ARMS = {
    "all":   "--pseudo_scope all",
    "lin10": "--pseudo_scope topk --pseudo_k 1,2,3,4,5,6,7,8,9,10",
    "agg50": "--pseudo_scope topk --pseudo_k 1,2,3,5,8,13,20,30,40,50",
}
GRID = dict(al=["bayes_product"], seed=[1, 2, 3])
FIXED_ARGS = (
    "--track_progress "
    "--train_label_mode pseudo "
    "--eval_model_type from_features "
    "--kernel_type tophat "
    "--budget 100 "
    "--initial_delta 0.75 "
    "--max_iter 9 "          # 100 -> 1000 labeled set (10 episodes)
)

ARRAY_THROTTLE = 6  # max concurrent array tasks (%N) — keep within group GPU quota
SWEEP_SCRIPT = Path(__file__).resolve().parent / "_bayes_pseudo_sweep.sh"


def al_configs(smoke=False):
    cfgs = []
    noise_cfgs = NOISE_CFGS if not smoke else [NOISE_CFGS[2]]  # smoke: noise 0.5 only
    for cfg_file, noise in noise_cfgs:
        for arm, arm_args in ARMS.items():
            cfgs.append(f"--cfg {CFG_DIR}/{cfg_file} --tag noise={noise} "
                        f"{arm_args} --tag pseudo={arm}")
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="3-job smoke test: one per arm, noise 0.5, seed 1, "
                         "max_iter 1, no --track_progress.")
    ap.add_argument("--submit", action="store_true",
                    help="Write the .sh and sbatch it via the SLURM login node.")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.smoke:
        grid = dict(al=["bayes_product"], seed=[1])
        fixed = FIXED_ARGS.replace("--max_iter 9", "--max_iter 1") \
                          .replace("--track_progress ", "")
        sweep_name = f"BAYESCOV_pseudo_smoke__{ts}"
    else:
        grid = GRID
        fixed = FIXED_ARGS
        sweep_name = f"BAYESCOV_pseudo_cifar100_n0_20_50_80__{ts}"
    cfgs = al_configs(smoke=args.smoke)
    jobs = build_jobs_from(cfgs, grid, fixed, sweep_name=sweep_name)
    script = build_script(jobs, ARRAY_THROTTLE)

    print(f"Sweep name : {sweep_name}")
    print(f"Arms       : {list(ARMS)}")
    print(f"Grid       : {grid}")
    print(f"Fixed args : {fixed.strip()}")
    print(f"Total runs : {len(jobs)}  (array 0-{len(jobs) - 1}%{ARRAY_THROTTLE})\n")
    for j in jobs:
        print("  " + j.replace(str(REPO_DIR) + "/", ""))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_SCRIPT.write_text(script)
    print(f"\nWritten: {SWEEP_SCRIPT}")

    if not args.submit:
        print("\nDry-run. Re-run with --submit to launch "
              "(CHECK GROUP GPU QUOTA FIRST — see daphna_lab/CLAUDE.md).")
        return

    submit_cmd = f"SLURM_CONF={SLURM_CONF} {SBATCH_BIN} {SWEEP_SCRIPT}"
    ssh_cmd = ["ssh", SLURM_LOGIN_NODE, submit_cmd]
    print(f"\nSubmitting: {' '.join(ssh_cmd)}")
    result = subprocess.run(ssh_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Submitted: {result.stdout.strip()}")
    else:
        print(f"ERROR (rc={result.returncode}): {result.stderr.strip()}")


if __name__ == "__main__":
    main()
