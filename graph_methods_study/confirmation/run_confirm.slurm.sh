#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
set -euo pipefail
cd "$1"
p="$(dirname "$PWD")"
export PYTHONPATH="$p/cifar_signed_audit_grid_20261126" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python="$p/daphna_lab/.venv-vast/bin/python"
printf 'Confirming seed %s on %s\n' "$SLURM_ARRAY_TASK_ID" "$(hostname)"
"$python" -u evaluate.py "$SLURM_ARRAY_TASK_ID" "$p/cifar_active_audit_20261126/geometry.npz" \
  "$p/cifar_graph_depth_20261126/human.npy" "$p/cifar_exact_backup_20261126/orders.json" \
  "seed_${SLURM_ARRAY_TASK_ID}.json"
