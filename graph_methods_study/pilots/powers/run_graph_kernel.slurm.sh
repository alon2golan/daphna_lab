#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
set -euo pipefail

cd "$1"
export PYTHONPATH=/cs/labs/daphna/tomn/projects/cifar_active_audit_20261126/human
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONDONTWRITEBYTECODE=1
python=/cs/labs/daphna/tomn/projects/daphna_lab/.venv-vast/bin/python
"$python" -m unittest -v
"$python" -u graph_kernel.py "$SLURM_ARRAY_TASK_ID" /cs/labs/daphna/tomn/projects/cifar_active_audit_20261126/geometry.npz "seed_${SLURM_ARRAY_TASK_ID}.json"
