#!/bin/bash
# Sweep the fine:coarse label-cost RATIO to find the break-even for the coarse task.
#
# run_coarse_budget.sbatch fixes one ratio; this submits one job per ratio, each into its
# own wandb group (coarse_ratio_v1_r<N>) so the 8-cell grid's run names never collide across
# ratios. Read each group's test_coarse_accuracy vs annotation_cost panel: the ratio at which
# the fine-rolled-up curve stops beating coarse-direct is the break-even.
#
# Direction: at ratio 1 fine and coarse buy the SAME number of points, so fine wins (its labels
# also carry the coarse info); as the ratio grows, coarse buys ratio x more points and takes over.
# The crossover therefore sits in 1..5 -- higher ratios only widen coarse's lead, so we stop at 5.
#
# Usage:
#   bash hierarchical_framework/scripts/launch_coarse_ratio_sweep.sh <EMB_DIR> [DELTA] [BUDGET] [ROUNDS] [BASE_GROUP] [PROJECT] [RATIOS...]
#
# Example (defaults: delta=auto, budget=100, rounds=6, base_group=coarse_ratio_v1, ratios 1 2 3 4 5):
#   bash hierarchical_framework/scripts/launch_coarse_ratio_sweep.sh \
#       hierarchical_framework/embeddings/cifar100/simclr/new_solo_learn

set -e

EMB_DIR=${1:?usage: launch_coarse_ratio_sweep.sh EMB_DIR [DELTA] [BUDGET] [ROUNDS] [BASE_GROUP] [PROJECT] [RATIOS...]}
DELTA=${2:-auto_delta}
BUDGET=${3:-100}
ROUNDS=${4:-6}
BASE_GROUP=${5:-coarse_ratio_v1}
PROJECT=${6:-superclass_analysis}
shift $(( $# < 6 ? $# : 6 )) || true
RATIOS=("$@")
[ ${#RATIOS[@]} -eq 0 ] && RATIOS=(1 2 3 4 5)   # default bracket around the crossover

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
SBATCH="$ROOT/hierarchical_framework/scripts/run_coarse_budget.sbatch"

echo "ratio sweep: emb=$EMB_DIR delta=$DELTA budget=$BUDGET rounds=$ROUNDS ratios=${RATIOS[*]} base_group=$BASE_GROUP project=$PROJECT"

# One job per ratio -> isolated group. run_coarse_budget.sbatch runs the full {probcover,random} x
# {1nn,mlp} x {fine,coarse} grid at that ratio, so each group is a self-contained equal-spend compare.
for RATIO in "${RATIOS[@]}"; do
  GROUP="${BASE_GROUP}_r${RATIO}"
  echo "--- submitting ratio=$RATIO -> group=$GROUP ---"
  sbatch "$SBATCH" "$EMB_DIR" "$DELTA" "$BUDGET" "$ROUNDS" "$RATIO" "$GROUP" "$PROJECT"
done

echo "submitted ${#RATIOS[@]} jobs."
