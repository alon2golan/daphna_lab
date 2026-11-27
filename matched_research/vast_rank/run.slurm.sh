#!/bin/bash
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:25:00
#SBATCH --job-name=vast-rank-oof
set -euo pipefail
cd "$1"
phase="$2"
p="$(dirname "$PWD")"
reference="$p/cifar100_balanced_depths_20261123"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH="$PWD:$PWD/helpers:$p/cifar_source_support_20261126_v2/source:$reference/source:$p/vast_followup_20261024/runtime_deps:$p/vast_expanded_20261019/runtime_deps"
export MPLCONFIGDIR="$PWD/matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"
python="$p/daphna_lab/.venv-vast/bin/python"
if [ "$phase" = oof ]; then
  "$python" -u rank.py "$SLURM_ARRAY_TASK_ID" "$reference/study" \
    "$p/cifar_vast_root_targets_20261127/results" targets
elif [ "$phase" = fit ]; then
  "$python" -u fit.py "$SLURM_ARRAY_TASK_ID" "$reference/study" targets fits
else
  exit 2
fi
