#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --job-name=cifar-grouped-backup
set -euo pipefail
cd "$1"
projects=/cs/labs/daphna/tomn/projects
export PYTHONPATH="$projects/cifar_active_audit_20261126/human"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONDONTWRITEBYTECODE=1
seed=$((70 + SLURM_ARRAY_TASK_ID))
python="$projects/daphna_lab/.venv-vast/bin/python"
"$python" -m unittest -q test_grouped_backup
"$python" -u grouped_backup.py --seed "$seed" \
  --geometry "$projects/cifar_active_audit_20261126/geometry.npz" --output "seed_${seed}.json"
