"""
Launch the §7.9 out-of-ball propagation sweep: pseudo_all + OUT_OF_BALL=nw
(noise-aware soft-kNN targets for uncovered pool points) on NORMALIZED
classifier inputs (takeaway 9's recommended pseudo_all configuration).

Baseline for comparison: the pseudo_all rows of the NORMALIZED pseudo sweep
(BAYESCOV_norm_pseudo_cifar100_..., job 31048960) — same seeds, same configs,
and OUT_OF_BALL only adds targets for points guard (a) previously dropped
(covered points' targets unchanged, selection untouched) => per-seed paired;
the delta isolates the out-of-ball propagation.

Grid: noise {0.0,0.2,0.5,0.8} x seed {1,2,3} = 12 runs. Expected mechanism:
@100 labels the pseudo training set grows ~9k (covered ~20% of pool) -> full
pool (~45k), the far 80% receiving noise-aware-1-NN-quality targets
(probcover+1nn evaluates 38.1 @100 clean on this signal).

Usage:
    python batch_utils/sweep_bayes_pseudo_nw.py             # dry-run
    python batch_utils/sweep_bayes_pseudo_nw.py --smoke     # 1-job, 1-episode
    python batch_utils/sweep_bayes_pseudo_nw.py [--smoke] --submit
"""
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

from sweep_train_al import (
    build_jobs_from,
    SLURM_LOGIN_NODE,
    SBATCH_BIN,
    SLURM_CONF,
    REPO_DIR,
    LOG_DIR,
)
from sweep_bayes_coverage import build_script
from sweep_bayes_pseudo import NOISE_CFGS, CFG_DIR

FIXED_ARGS = (
    "--track_progress "
    "--train_label_mode pseudo "
    "--pseudo_scope all "
    "--out_of_ball nw "
    "--normalize_features "
    "--eval_model_type from_features "
    "--kernel_type tophat "
    "--budget 100 "
    "--initial_delta 0.75 "
    "--max_iter 9 "          # 100 -> 1000 labeled set (10 episodes)
    "--tag pseudo=all --tag oob=nw --tag feat_norm=l2 "
)

ARRAY_THROTTLE = 6
SWEEP_SCRIPT = Path(__file__).resolve().parent / "_bayes_pseudo_nw_sweep.sh"


def al_configs(smoke=False):
    noise_cfgs = NOISE_CFGS if not smoke else [NOISE_CFGS[2]]  # smoke: noise .5
    return [f"--cfg {CFG_DIR}/{cfg_file} --tag noise={noise}"
            for cfg_file, noise in noise_cfgs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1-job smoke: noise 0.5, seed 1, 1 episode, no --track_progress.")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.smoke:
        grid = dict(al=["bayes_product"], seed=[1])
        fixed = FIXED_ARGS.replace("--max_iter 9", "--max_iter 1") \
                          .replace("--track_progress ", "")
        sweep_name = f"BAYESCOV_nw_pseudo_smoke__{ts}"
    else:
        grid = dict(al=["bayes_product"], seed=[1, 2, 3])
        fixed = FIXED_ARGS
        sweep_name = f"BAYESCOV_nw_pseudo_cifar100_n0_20_50_80__{ts}"
    jobs = build_jobs_from(al_configs(smoke=args.smoke), grid, fixed,
                           sweep_name=sweep_name)
    script = build_script(jobs, ARRAY_THROTTLE)

    print(f"Sweep name : {sweep_name}")
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
