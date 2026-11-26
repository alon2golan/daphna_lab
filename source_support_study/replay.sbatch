#!/bin/bash
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --job-name=cifar-source-replay

set -euo pipefail
root="${1:?Study directory required}"
shift
projects="$(dirname "$root")"
reference="$projects/cifar100_balanced_depths_20261123"
python="$projects/daphna_lab/.venv-vast/bin/python"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$root/source:$reference/source:$projects/vast_followup_20261024/runtime_deps:$projects/vast_expanded_20261019/runtime_deps"
cd "$root/source"
printf 'Starting replay on %s\n' "$(hostname)"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    budgets=(200 500 1000 2000)
    set -- "$@" --budget "${budgets[SLURM_ARRAY_TASK_ID]}"
fi
"$python" -u -m source_support_study.replay --archive "$reference/study" "$@"
