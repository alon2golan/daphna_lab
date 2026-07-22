#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=8g
#SBATCH -c 2
#SBATCH --time=0:20:00
#SBATCH --output=hierarchical_framework/outputs/slurm_logs/coarse_merge_%j.out
#SBATCH --error=hierarchical_framework/outputs/slurm_logs/coarse_merge_%j.err
#SBATCH --killable
#SBATCH --requeue

# Optional post-step for the parallel array sweep: combine the per-shard CSVs into ONE clean
# CSV + PNG (and, with --wandb, one consolidated 'merged' run). CPU only -- no GPU needed.
# The LIVE view during the sweep is the overlaid wandb group; this is just the tidy final artifact.
#
# Usage (after the array job finished):
#   sbatch hierarchical_framework/scripts/run_coarse_budget_merge.sbatch --seed 0
#   sbatch hierarchical_framework/scripts/run_coarse_budget_merge.sbatch --selector random --seed 0 --wandb
#
# To chain it automatically after the array, submit with a dependency:
#   AID=$(sbatch --parsable hierarchical_framework/scripts/run_coarse_budget_sweep_array.sbatch)
#   sbatch --dependency=afterok:$AID hierarchical_framework/scripts/run_coarse_budget_merge.sbatch --seed 0

set -e

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
cd "$ROOT"
source "$ROOT/venv/bin/activate"
mkdir -p hierarchical_framework/outputs/slurm_logs

echo "merging coarse-budget shards: args=[$*]"

python -u hierarchical_framework/pipeline/finetune/run_coarse_budget_merge.py "$@"

echo "merge done."
