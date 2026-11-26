#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --job-name=cifar-backup-confirm
set -euo pipefail
cd "$1"
projects=/cs/labs/daphna/tomn/projects
export PYTHONPATH="$projects/cifar_active_audit_20261126/human"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONDONTWRITEBYTECODE=1
seeds=(30 31 32 33 34)
seed="${seeds[$SLURM_ARRAY_TASK_ID]}"
"$projects/daphna_lab/.venv-vast/bin/python" -u backup_coverage.py --seed "$seed" \
  --geometry "$projects/cifar_active_audit_20261126/geometry.npz" --output "seed_${seed}.json"
