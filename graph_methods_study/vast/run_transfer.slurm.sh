#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
set -euo pipefail
cd "$1"
p="$(dirname "$PWD")"
reference="$p/cifar100_balanced_depths_20261123"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$p/cifar_source_support_20261126_v2/source:$reference/source:$p/vast_followup_20261024/runtime_deps:$p/vast_expanded_20261019/runtime_deps"
"$p/daphna_lab/.venv-vast/bin/python" -u transfer.py "$SLURM_ARRAY_TASK_ID" "$reference/study" results
