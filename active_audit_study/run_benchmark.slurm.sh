#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
set -euo pipefail

cd "$1"
python=/cs/labs/daphna/tomn/projects/daphna_lab/.venv-vast/bin/python
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONDONTWRITEBYTECODE=1
printf 'Audit replay seed %s on %s\n' "$SLURM_ARRAY_TASK_ID" "$(hostname)"
"$python" -u benchmark.py run "$SLURM_ARRAY_TASK_ID" "$2" "seed_${SLURM_ARRAY_TASK_ID}.json"
