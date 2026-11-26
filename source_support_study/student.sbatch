#!/bin/bash
#SBATCH --partition=short
#SBATCH --gres=gg:g0:1,vmem:8g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=cifar-source-student

set -euo pipefail
root="${1:?Study directory required}"
output="${2:?Output directory required}"
epochs="${3:-300}"
projects="$(dirname "$root")"
reference="$projects/cifar100_balanced_depths_20261123"
python="$projects/daphna_lab/.venv-vast/bin/python"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH="$root/source:$reference/source:$projects/vast_followup_20261024/runtime_deps:$projects/vast_expanded_20261019/runtime_deps"
export MPLCONFIGDIR="$output/matplotlib_cache"
cd "$root/source"
mkdir -p "$MPLCONFIGDIR"
printf 'Starting student fit on %s\n' "$(hostname)"
"$python" -u -m source_support_study.student --archive "$reference/study" \
  --output "$output" --index "${SLURM_ARRAY_TASK_ID:-0}" --epochs "$epochs"
