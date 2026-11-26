#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
set -euo pipefail
cd "$1"
projects="$(dirname "$PWD")"
audit="$projects/cifar_signed_audit_grid_20261126"
python="$projects/daphna_lab/.venv-vast/bin/python"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$audit"
printf 'Depth replay seed %s on %s\n' "$SLURM_ARRAY_TASK_ID" "$(hostname)"
"$python" -u depth.py "$projects/cifar_active_audit_20261126/geometry.npz" \
  "$audit/seed_${SLURM_ARRAY_TASK_ID}.json" human.npy "$SLURM_ARRAY_TASK_ID" "seed_${SLURM_ARRAY_TASK_ID}.json"
