#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=32g
#SBATCH -c 4
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=hierarchical_framework/outputs/slurm_logs/coarse_budget_%j.out
#SBATCH --error=hierarchical_framework/outputs/slurm_logs/coarse_budget_%j.err
#SBATCH --killable
#SBATCH --requeue

# Coarse-vs-fine labeling experiment: for the COARSE (superclass) task, is the AL win
# (probcover vs random) and the labeling granularity (few fine labels vs many cheap coarse
# labels) different? Sweeps every combination of:
#   selection    : probcover, random
#   evaluator    : 1nn, mlp        (flat classifiers only -- no hierarchical heads)
#   granularity  : fine, coarse    (fine also logs its rolled-up coarse curve via --coarse)
# All 8 runs land in one fresh wandb group so they don't collide with the old sweeps; compare
# them on the test_coarse_accuracy panel (fine runs mirror their coarse curve there too).
#
# Usage:
#   sbatch hierarchical_framework/scripts/run_coarse_budget.sbatch <EMB_DIR> [DELTA] [BUDGET] [ROUNDS] [GROUP] [PROJECT]
#
# Positional args:
#   1) EMB_DIR    embeddings dir with <split>_features_normalized.npy + labels + test_*
#   2) DELTA      float | knee_delta | auto_delta   (default auto_delta)
#   3) BUDGET     points per round                  (default 100)
#   4) ROUNDS     number of AL rounds               (default 6)
#   5) GROUP      wandb group                       (default coarse_budget_v1)
#   6) PROJECT    wandb project                     (default superclass_analysis)
#
# Example:
#   sbatch hierarchical_framework/scripts/run_coarse_budget.sbatch \
#       hierarchical_framework/embeddings/cifar100/simclr/new_solo_learn

set -e

EMB_DIR=${1:?usage: sbatch run_coarse_budget.sbatch EMB_DIR [DELTA] [BUDGET] [ROUNDS] [GROUP] [PROJECT]}
DELTA=${2:-auto_delta}
BUDGET=${3:-100}
ROUNDS=${4:-6}
GROUP=${5:-coarse_budget_v1}           # fresh group -> isolated from the old runs
PROJECT=${6:-superclass_analysis}

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
cd "$ROOT"
# venv has the full stack AND wandb (.venv_clean does not)
source "$ROOT/venv/bin/activate"
mkdir -p hierarchical_framework/outputs/slurm_logs

echo "coarse-budget sweep: emb=$EMB_DIR delta=$DELTA budget=$BUDGET rounds=$ROUNDS group=$GROUP project=$PROJECT"

# The 2 x 2 x 2 grid. Fine runs add --coarse so their rolled-up superclass curve is logged too.
for STRATEGY in probcover random; do
  for EVAL in 1nn mlp; do
    for GRAN in fine coarse; do
      COARSE_FLAG=""
      [ "$GRAN" = "fine" ] && COARSE_FLAG="--coarse"   # fine: also log the rolled-up coarse curve
      echo "--- $STRATEGY / $EVAL / $GRAN ---"
      python -u hierarchical_framework/pipeline/run_active_learning.py \
        --embeddings_dir "$EMB_DIR" \
        --strategy "$STRATEGY" \
        --eval "$EVAL" \
        --label_granularity "$GRAN" $COARSE_FLAG \
        --delta "$DELTA" \
        --budget "$BUDGET" \
        --rounds "$ROUNDS" \
        --wandb --project "$PROJECT" --group "$GROUP"
    done
  done
done

echo "done."
