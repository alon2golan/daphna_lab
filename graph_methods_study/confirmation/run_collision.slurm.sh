#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
set -euo pipefail
cd "$1"
p="$(dirname "$PWD")"
confirm="$p/cifar_graph_methods_confirmation_20261127"
export PYTHONPATH="$p/cifar_signed_audit_grid_20261126:$p/cifar_source_support_20261126_v2/source:$confirm"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
"$p/daphna_lab/.venv-vast/bin/python" -u collision.py "$SLURM_ARRAY_TASK_ID" \
    "$p/cifar_active_audit_20261126/geometry.npz" "$confirm/seed_${SLURM_ARRAY_TASK_ID}.json" \
    "seed_${SLURM_ARRAY_TASK_ID}.json"
