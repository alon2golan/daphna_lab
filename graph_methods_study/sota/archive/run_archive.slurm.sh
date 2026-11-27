#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
set -euo pipefail
cd "$1"
p="$(dirname "$PWD")"
export PYTHONPATH="$p/cifar_graph_methods_confirmation_20261127:$p/cifar_signed_audit_grid_20261126"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python="$p/daphna_lab/.venv-vast/bin/python"
"$python" -m unittest -v
"$python" -u replay.py "$SLURM_ARRAY_TASK_ID" "$p" "condition_${SLURM_ARRAY_TASK_ID}.json"
