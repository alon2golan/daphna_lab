#!/bin/bash
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --job-name=cifar-source-channel

set -euo pipefail
root="${1:?Study directory required}"
projects="$(dirname "$root")"
reference="$projects/cifar100_balanced_depths_20261123"
python="$projects/daphna_lab/.venv-vast/bin/python"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$root/source:$reference/source:$projects/vast_followup_20261024/runtime_deps:$projects/vast_expanded_20261019/runtime_deps"
cd "$root/source"
"$python" -u -m source_support_study.channel --projects "$projects" \
    --audit "$projects/cifar_active_audit_20261126" \
    --output "$root/results/channel_${SLURM_ARRAY_TASK_ID}.json" --seed "$SLURM_ARRAY_TASK_ID"
