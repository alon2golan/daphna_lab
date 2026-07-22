#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=32g
#SBATCH -c 4
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=hierarchical_framework/outputs/slurm_logs/pc_flat_sweep_%j.out
#SBATCH --error=hierarchical_framework/outputs/slurm_logs/pc_flat_sweep_%j.err
#SBATCH --killable
#SBATCH --requeue

# ProbCover selector scored by the two FLAT evaluators (1-NN and MLP) over a growing
# budget 100, 200, 300, ..., 2000. The runner grows the labeled set by BUDGET points
# each round, so BUDGET=100 x ROUNDS=20 lands one point at every budget in that range.
# Each evaluator is its own live wandb run; both share the SAME project + a NEW group,
# so the two curves sit side by side. wandb streams per round (live) AND the run files
# are written under outputs/wandb (WANDB_DIR below), so nothing is lost off-cluster.
#
# Usage:
#   sbatch hierarchical_framework/scripts/run_probcover_flat_budget_sweep.sbatch
#   sbatch hierarchical_framework/scripts/run_probcover_flat_budget_sweep.sbatch <EMB_DIR> <GROUP> <PROJECT> <DELTA>
#
# Positional args (all optional):
#   1) EMB_DIR   embeddings dir            (default cifar100 simclr new_solo_learn)
#   2) GROUP     wandb group (the NEW one) (default probcover_flat_budget_100_2000)
#   3) PROJECT   wandb project (SAME one)  (default superclass_analysis)
#   4) DELTA     float | knee_delta | auto_delta   (default auto_delta)

set -e

EMB_DIR=${1:-hierarchical_framework/embeddings/cifar100/simclr/new_solo_learn}
GROUP=${2:-probcover_flat_budget_100_2000}
PROJECT=${3:-superclass_analysis}
DELTA=${4:-auto_delta}

BUDGET=100    # points added per round
ROUNDS=20     # 100, 200, ..., 2000

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
cd "$ROOT"
# venv has the full stack AND wandb (.venv_clean does not)
source "$ROOT/venv/bin/activate"
mkdir -p hierarchical_framework/outputs/slurm_logs

# Keep wandb's local run files under outputs/ (not ROOT/wandb) so they're saved with the rest.
export WANDB_DIR="$ROOT/hierarchical_framework/outputs"

echo "ProbCover flat sweep: emb=$EMB_DIR group=$GROUP project=$PROJECT delta=$DELTA budget=$BUDGET rounds=$ROUNDS"

# One live wandb run per flat evaluator, both in project=$PROJECT group=$GROUP.
for EVAL in 1nn mlp; do
  echo "=== eval=$EVAL ==="
  python -u hierarchical_framework/pipeline/run_active_learning.py \
    --embeddings_dir "$EMB_DIR" \
    --strategy probcover \
    --eval "$EVAL" \
    --delta "$DELTA" \
    --budget "$BUDGET" \
    --rounds "$ROUNDS" \
    --wandb --project "$PROJECT" --group "$GROUP"
done

echo "done."
