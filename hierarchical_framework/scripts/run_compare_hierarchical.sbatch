#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=32g
#SBATCH -c 4
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=hierarchical_framework/outputs/slurm_logs/compare_%j.out
#SBATCH --error=hierarchical_framework/outputs/slurm_logs/compare_%j.err
#SBATCH --killable
#SBATCH --requeue

# Compare every hierarchical evaluator across selection strategies (probcover / random / all).
# Trains MLP + kNN heads on CPU (sklearn); the GPU only speeds up ProbCover's delta-graph build.
# Writes a figure + CSV under outputs/figures/hierarchical_comparison/; does NOT use wandb.
#
# Usage:
#   sbatch hierarchical_framework/scripts/run_compare_hierarchical.sbatch <EMB_DIR> [DELTA] [BUDGET] [ROUNDS] [PROJECT]
#
# Positional args:
#   1) EMB_DIR   embeddings dir with <split>_features_normalized.npy + labels + test_*
#   2) DELTA     float | knee_delta | auto_delta   (default auto_delta: pinned table -> knee)
#   3) BUDGET    points per round                  (default 100)
#   4) ROUNDS    rounds for the growing strategies (default 6)
#   5) PROJECT   wandb project                     (default superclass_analysis)
#
# Example (new = solo-learn; swap new_solo_learn -> old_deep_al for itai's baseline):
#   sbatch hierarchical_framework/scripts/run_compare_hierarchical.sbatch \
#       hierarchical_framework/embeddings/cifar100/simclr/new_solo_learn

set -e

EMB_DIR=${1:?usage: sbatch run_compare_hierarchical.sbatch EMB_DIR [DELTA] [BUDGET] [ROUNDS] [PROJECT]}
DELTA=${2:-auto_delta}
BUDGET=${3:-100}
ROUNDS=${4:-6}
PROJECT=${5:-superclass_analysis}

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
cd "$ROOT"
# venv has the full stack (torch + sklearn + matplotlib + deep_al deps); .venv_clean lacks some
source "$ROOT/venv/bin/activate"
mkdir -p hierarchical_framework/outputs/slurm_logs

echo "COMPARE: emb=$EMB_DIR delta=$DELTA budget=$BUDGET rounds=$ROUNDS project=$PROJECT"

python -u hierarchical_framework/pipeline/compare_hierarchical.py \
  --embeddings_dir "$EMB_DIR" \
  --delta "$DELTA" \
  --budget "$BUDGET" \
  --rounds "$ROUNDS" \
  --wandb --project "$PROJECT"

echo "done."
