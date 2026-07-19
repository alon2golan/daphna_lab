#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=32g
#SBATCH -c 4
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=hierarchical_framework/outputs/slurm_logs/run_al_%j.out
#SBATCH --error=hierarchical_framework/outputs/slurm_logs/run_al_%j.err
#SBATCH --killable
#SBATCH --requeue

# Run the AL pipeline (ProbCover or another strategy) with 1-NN eval + wandb.
#
# Usage:
#   sbatch hierarchical_framework/scripts/run_active_learning.sbatch <EMB_DIR> [STRATEGY] [DELTA] [BUDGET] [ROUNDS] [PROJECT]
#
# Positional args:
#   1) EMB_DIR    embeddings dir with <split>_features_normalized.npy + labels + test_*
#   2) STRATEGY   probcover | random   (default probcover)
#   3) DELTA      coverage radius       (default 0.65)
#   4) BUDGET     points per round      (default 100)
#   5) ROUNDS     number of AL rounds   (default 10)
#   6) PROJECT    wandb project         (default bayesian)
#
# Example:
#   sbatch hierarchical_framework/scripts/run_active_learning.sbatch \
#       solo-learn/embeddings/simclr/cifar100/1000ep probcover 0.65 100 10

set -e

EMB_DIR=${1:?usage: sbatch run_active_learning.sbatch EMB_DIR [STRATEGY] [DELTA] [BUDGET] [ROUNDS] [PROJECT]}
STRATEGY=${2:-probcover}
DELTA=${3:-0.65}
BUDGET=${4:-100}
ROUNDS=${5:-10}
PROJECT=${6:-bayesian}      # wandb project (6th arg); default keeps old behaviour

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
cd "$ROOT"
# venv has the full stack AND wandb (.venv_clean does not)
source "$ROOT/venv/bin/activate"
mkdir -p hierarchical_framework/outputs/slurm_logs

echo "AL: strategy=$STRATEGY emb=$EMB_DIR delta=$DELTA budget=$BUDGET rounds=$ROUNDS project=$PROJECT"

python -u hierarchical_framework/pipeline/run_active_learning.py \
  --embeddings_dir "$EMB_DIR" \
  --strategy "$STRATEGY" \
  --delta "$DELTA" \
  --budget "$BUDGET" \
  --rounds "$ROUNDS" \
  --wandb --project "$PROJECT"

echo "done."
