"""
Normalized-features rerun of the three sweeps behind the bayesian-coverage doc
(docs/bayesian_coverage_RESULTS.md §4 tables) — validates every classifier-trained
number with L2-normalized inputs (--normalize_features).

WHY: the from_features classifier historically trained on RAW SimCLR features
(custom_datasets.py served normalized=False) while the 1-NN eval arm was fixed to
L2-normalize (~+18 pts on CIFAR100 @100 labels). The AL-side kernel/posterior was
ALWAYS normalized, so selection is untouched — per-seed lSet trajectories are
identical to the historical runs and every before/after delta is attributable
purely to classifier input normalization. The probcover+1nn rows need NO rerun.

Three job families, ONE sbatch array (single %6 throttle so we never hold more
than ~6 of the group's 24 GPUs; three separate arrays at %6 each could hold 18):

  compare   48 = {random,probcover,bayes_additive,bayes_product} x noise{0,.2,.5,.8} x seed{1,2,3}
              (normalized twin of job 30978366, BAYESCOV_compare_cifar100_...)
  trainmode 60 = bayes_product x {observed,map_posterior,filtered,soft_posterior,reweighted}
              x noise x seed   (twin of job 30980646, BAYESCOV_trainmode_cifar100_...)
  pseudo    36 = bayes_product pseudo arms {all,lin10,agg50} x noise x seed
              (twin of job 31047129, BAYESCOV_pseudo_cifar100_...)

All jobs add `--normalize_features --tag feat_norm=l2`; result_writer records
params.normalize_features=True (param, not just tag -> distinct config_key from
the unnormalized originals). Sweep names carry the BAYESCOV token (auto-category
bayesian_coverage) + a `norm` token; compare with batch_utils/compare_norm_rerun.py.

Usage:
    python batch_utils/sweep_bayes_norm_rerun.py             # dry-run: print plan
    python batch_utils/sweep_bayes_norm_rerun.py --smoke     # 3-job, 1-episode smoke
    python batch_utils/sweep_bayes_norm_rerun.py [--smoke] --submit
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
from sweep_bayes_coverage import build_script, al_configs_for, DATASET_DIRS
from sweep_bayes_pseudo import ARMS as PSEUDO_ARMS, NOISE_CFGS, CFG_DIR

NORM_ARGS = "--normalize_features --tag feat_norm=l2 "
FIXED_ARGS = (
    "--track_progress "
    "--eval_model_type from_features "
    "--kernel_type tophat "
    "--budget 100 "
    "--initial_delta 0.75 "
    "--max_iter 9 "          # 100 -> 1000 labeled set (10 episodes)
) + NORM_ARGS
TRAIN_LABEL_MODES = ["observed", "map_posterior", "filtered", "soft_posterior", "reweighted"]

ARRAY_THROTTLE = 6  # single throttle across ALL 144 jobs — group-quota etiquette
SWEEP_SCRIPT = Path(__file__).resolve().parent / "_bayes_norm_rerun_sweep.sh"


def pseudo_al_configs():
    cfgs = []
    for cfg_file, noise in NOISE_CFGS:
        for arm, arm_args in PSEUDO_ARMS.items():
            cfgs.append(f"--cfg {CFG_DIR}/{cfg_file} --tag noise={noise} "
                        f"{arm_args} --tag pseudo={arm}")
    return cfgs


def build_all_jobs(ts, smoke=False):
    """Three (family, sweep_name, jobs) triples; smoke = 1 job per family, 1 episode."""
    cc = al_configs_for(DATASET_DIRS["cifar100"])
    fixed = FIXED_ARGS if not smoke else \
        FIXED_ARGS.replace("--max_iter 9", "--max_iter 1").replace("--track_progress ", "")
    seeds = [1, 2, 3] if not smoke else [1]

    families = [
        ("compare",
         f"BAYESCOV_norm_compare_cifar100_n0_20_50_80__{ts}",
         cc if not smoke else [cc[2]],                      # smoke: noise 0.5 only
         dict(al=["random", "probcover", "bayes_additive", "bayes_product"] if not smoke
              else ["bayes_product"], seed=seeds),
         ""),
        ("trainmode",
         f"BAYESCOV_norm_trainmode_cifar100_n0_20_50_80__{ts}",
         cc if not smoke else [cc[2]],
         dict(al=["bayes_product"],
              train_label_mode=TRAIN_LABEL_MODES if not smoke else ["soft_posterior"],
              seed=seeds),
         ""),
        ("pseudo",
         f"BAYESCOV_norm_pseudo_cifar100_n0_20_50_80__{ts}",
         pseudo_al_configs() if not smoke
         else [c for c in pseudo_al_configs() if "noise=0.5" in c and "pseudo=all" in c],
         dict(al=["bayes_product"], seed=seeds),
         "--train_label_mode pseudo "),
    ]
    out = []
    for name, sweep_name, cfgs, grid, extra in families:
        jobs = build_jobs_from(cfgs, grid, fixed + extra, sweep_name=sweep_name)
        out.append((name, sweep_name, jobs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="3-job smoke (one per family, noise 0.5, seed 1, 1 episode).")
    ap.add_argument("--submit", action="store_true",
                    help="Write the .sh and sbatch it via the SLURM login node.")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    families = build_all_jobs(ts, smoke=args.smoke)
    all_jobs = [j for _, _, jobs in families for j in jobs]
    script = build_script(all_jobs, ARRAY_THROTTLE)

    print(f"Normalized rerun — {len(all_jobs)} jobs, array 0-{len(all_jobs)-1}%{ARRAY_THROTTLE}")
    for name, sweep_name, jobs in families:
        print(f"  {name:<9} {len(jobs):>3} jobs  sweep_name={sweep_name}")
    print()
    for j in all_jobs:
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
