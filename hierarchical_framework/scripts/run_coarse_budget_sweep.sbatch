#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=48g
#SBATCH -c 8
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:l40s:2
#SBATCH --output=hierarchical_framework/outputs/slurm_logs/coarse_sweep_%j.out
#SBATCH --error=hierarchical_framework/outputs/slurm_logs/coarse_sweep_%j.err
#SBATCH --killable
#SBATCH --requeue

# Coarse-budget sweep: for each budget (points treated as coarse-labeled), compare a 20-way
# superclass MLP probe on the FROZEN SimCLR embedding against the same probe on a backbone
# FINETUNED with the coarse head only. Both curves use the same labeled points at each budget,
# so the only difference is whether the embedding moved. Writes a CSV + PNG under
# outputs/finetune/ and streams both curves to Weights & Biases (group coarse_budget_finetune).
#
# Usage:
#   sbatch hierarchical_framework/scripts/run_coarse_budget_sweep.sbatch
#   sbatch hierarchical_framework/scripts/run_coarse_budget_sweep.sbatch --seed 1
#   sbatch hierarchical_framework/scripts/run_coarse_budget_sweep.sbatch --selector random --budgets 100 500 1000
#
# Any extra args are forwarded straight to run_coarse_budget_sweep.py.

set -e

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
cd "$ROOT"
# venv has the full stack (torch + solo-learn + sklearn + matplotlib + wandb); .venv_clean lacks some
source "$ROOT/venv/bin/activate"
mkdir -p hierarchical_framework/outputs/slurm_logs

echo "coarse-budget sweep: forwarding args -> $*"

# --wandb on by default; drop it to only write the CSV + PNG. Extra CLI args ($@) pass through.
python -u hierarchical_framework/pipeline/finetune/run_coarse_budget_sweep.py --wandb "$@"

echo "done."
