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
#
# Cost model: a fine label costs COST_RATIO units, a coarse label costs 1. To compare at EQUAL
# SPEND, the coarse runs label COST_RATIO x more points per round, so both regimes sweep the same
# annotation_cost range (fine: BUDGET pts/round x COST_RATIO = coarse: BUDGET*COST_RATIO pts/round
# x 1). Read the result on the test_coarse_accuracy panel with x-axis = annotation_cost: that is
# few-fine vs many-cheap-coarse at matched budget. All 8 runs land in one fresh wandb group.
#
# Usage:
#   sbatch hierarchical_framework/scripts/run_coarse_budget.sbatch <EMB_DIR> [DELTA] [BUDGET] [ROUNDS] [COST_RATIO] [GROUP] [PROJECT]
#
# Positional args:
#   1) EMB_DIR    embeddings dir with <split>_features_normalized.npy + labels + test_*
#   2) DELTA      float | knee_delta | auto_delta   (default auto_delta)
#   3) BUDGET     fine points per round             (default 100; coarse uses BUDGET*COST_RATIO)
#   4) ROUNDS     number of AL rounds               (default 6)
#   5) COST_RATIO fine-label cost vs coarse=1       (default 5)
#   6) GROUP      wandb group                       (default coarse_budget_v1)
#   7) PROJECT    wandb project                     (default superclass_analysis)
#
# Example:
#   sbatch hierarchical_framework/scripts/run_coarse_budget.sbatch \
#       hierarchical_framework/embeddings/cifar100/simclr/new_solo_learn

set -e

EMB_DIR=${1:?usage: sbatch run_coarse_budget.sbatch EMB_DIR [DELTA] [BUDGET] [ROUNDS] [COST_RATIO] [GROUP] [PROJECT]}
DELTA=${2:-auto_delta}
BUDGET=${3:-100}
ROUNDS=${4:-6}
COST_RATIO=${5:-5}                     # fine label costs this many coarse-label units
GROUP=${6:-coarse_budget_v1}           # fresh group -> isolated from the old runs
PROJECT=${7:-superclass_analysis}

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
cd "$ROOT"
# venv has the full stack AND wandb (.venv_clean does not)
source "$ROOT/venv/bin/activate"
mkdir -p hierarchical_framework/outputs/slurm_logs

COARSE_BUDGET=$((BUDGET * COST_RATIO))   # coarse buys COST_RATIO x more points for the same spend

echo "coarse-budget sweep: emb=$EMB_DIR delta=$DELTA fine_budget=$BUDGET coarse_budget=$COARSE_BUDGET rounds=$ROUNDS cost_ratio=$COST_RATIO group=$GROUP project=$PROJECT"

# The 2 x 2 x 2 grid. Fine runs add --coarse so their rolled-up superclass curve is logged too.
# Coarse runs use COARSE_BUDGET points/round so both regimes span the same annotation_cost range.
for STRATEGY in probcover random; do
  for EVAL in 1nn mlp; do
    for GRAN in fine coarse; do
      if [ "$GRAN" = "fine" ]; then
        COARSE_FLAG="--coarse"           # fine: also log the rolled-up coarse curve
        ROUND_BUDGET="$BUDGET"
      else
        COARSE_FLAG=""
        ROUND_BUDGET="$COARSE_BUDGET"    # coarse: more points per round, matched spend
      fi
      echo "--- $STRATEGY / $EVAL / $GRAN (budget=$ROUND_BUDGET) ---"
      python -u hierarchical_framework/pipeline/run_active_learning.py \
        --embeddings_dir "$EMB_DIR" \
        --strategy "$STRATEGY" \
        --eval "$EVAL" \
        --label_granularity "$GRAN" $COARSE_FLAG \
        --label_cost_ratio "$COST_RATIO" \
        --delta "$DELTA" \
        --budget "$ROUND_BUDGET" \
        --rounds "$ROUNDS" \
        --wandb --project "$PROJECT" --group "$GROUP"
    done
  done
done

echo "done."
