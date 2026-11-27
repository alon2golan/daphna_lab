#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail
cd "$1"
p="$(dirname "$PWD")"
reference="$p/cifar100_herding_20261119"
export PYTHONPATH="$p/cifar_signed_audit_grid_20261126:$p/cifar_graph_methods_confirmation_20261127:$reference/source:$p/vast_followup_20261024/runtime_deps:$p/vast_expanded_20261019/runtime_deps"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
"$p/daphna_lab/.venv-vast/bin/python" -u fragility.py "$SLURM_ARRAY_TASK_ID" "$reference/study" \
  "$p/cifar_active_audit_20261126/geometry.npz" "$p/cifar_exact_backup_20261126/orders.json" \
  "$p/cifar_graph_depth_20261126/human.npy" "case_${SLURM_ARRAY_TASK_ID}.json"
