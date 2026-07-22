#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=32g
#SBATCH -c 4
#SBATCH --time=6:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --array=0-5
#SBATCH --output=hierarchical_framework/outputs/slurm_logs/coarse_sweep_%A_%a.out
#SBATCH --error=hierarchical_framework/outputs/slurm_logs/coarse_sweep_%A_%a.err
#SBATCH --killable
#SBATCH --requeue

# PARALLEL coarse-budget sweep as a slurm job ARRAY: each array task takes ONE GPU from the gg:g0
# group (a5000 / rtx2080 / l4 -- any type; tasks schedule independently, so you don't need many
# GPUs at once and mixed hardware is fine). For the l40s nodes use gg:g4 instead. Each task runs
# one shard of the budgets. Every task is its OWN LIVE wandb run in the SAME group with
# the SAME metric keys, so overlaying the group's runs on one panel shows the two curves filling
# in LIVE as each task reports -- no waiting for a merge. Each task also writes its own shard CSV.
#
# The budget shards below split the work so the expensive large budgets sit alone (they are the
# long pole and cannot be split), while the cheap small budgets share one fast task. Adjust the
# array range (0-N) to match the number of shards.
#
# Usage:
#   sbatch hierarchical_framework/scripts/run_coarse_budget_sweep_array.sbatch
#   sbatch hierarchical_framework/scripts/run_coarse_budget_sweep_array.sbatch --seed 1
#   (extra args after the script are forwarded to every task, e.g. --selector random)
#
# After it finishes, optionally build the single combined CSV + PNG:
#   sbatch hierarchical_framework/scripts/run_coarse_budget_merge.sbatch --seed 0

set -e

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
cd "$ROOT"
source "$ROOT/venv/bin/activate"
mkdir -p hierarchical_framework/outputs/slurm_logs

# One entry per array index: the budgets that shard computes. Keep in sync with --array above.
BUDGET_SHARDS=(
  "100 200 300 400 500 600 700 800 900 1000"   # shard 0: all the cheap budgets, one fast task
  "2000 3000 4000"                              # shard 1
  "5000 6000 7000"                              # shard 2
  "8000 9000 10000"                             # shard 3
  "25000"                                       # shard 4
  "50000"                                       # shard 5: the long pole (full train set)
)

SHARD_BUDGETS=${BUDGET_SHARDS[$SLURM_ARRAY_TASK_ID]}
echo "shard $SLURM_ARRAY_TASK_ID budgets=[$SHARD_BUDGETS] extra args=[$*]"

# --wandb on so this shard streams its own live run; --shard names its csv + run uniquely.
python -u hierarchical_framework/pipeline/finetune/run_coarse_budget_sweep.py \
  --budgets $SHARD_BUDGETS \
  --shard "$SLURM_ARRAY_TASK_ID" \
  --wandb "$@"

echo "shard $SLURM_ARRAY_TASK_ID done."
