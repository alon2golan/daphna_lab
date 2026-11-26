#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
set -euo pipefail

cd "$1"
python=/cs/labs/daphna/tomn/projects/daphna_lab/.venv-vast/bin/python
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    "$python" -u active_audit.py replay "$SLURM_ARRAY_TASK_ID" geometry.npz "seed_${SLURM_ARRAY_TASK_ID}.json"
else
    "$python" -m unittest -v
    "$python" -u active_audit.py prepare neighbors.npz geometry.npz
fi
