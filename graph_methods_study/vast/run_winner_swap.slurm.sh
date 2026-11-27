#!/bin/bash
#SBATCH --partition=short
#SBATCH --gres=gg:g0:1,vmem:8g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=cifar-winner-swap
set -euo pipefail
cd "$1"
p="$(dirname "$PWD")"
reference="$p/cifar100_balanced_depths_20261123"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH="$p/cifar_source_support_20261126_v2/source:$reference/source:$p/vast_followup_20261024/runtime_deps:$p/vast_expanded_20261019/runtime_deps"
export MPLCONFIGDIR="$PWD/matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"
"$p/daphna_lab/.venv-vast/bin/python" -u winner_swap.py "$SLURM_ARRAY_TASK_ID" "$reference/study" \
  "$p/cifar_vast_root_targets_20261127/results" targets outputs
