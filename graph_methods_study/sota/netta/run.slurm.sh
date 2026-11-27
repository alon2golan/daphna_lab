#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --job-name=cifar-netta-baselines
set -euo pipefail
cd "$1"
projects=/cs/labs/daphna/tomn/projects
export PYTHONPATH="$projects/vast_followup_20261024/source:$projects/cifar_active_audit_20261126/human:$projects/vast_followup_20261024/runtime_deps:$projects/vast_expanded_20261019/runtime_deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=2 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
python="$projects/daphna_lab/.venv-vast/bin/python"
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    "$python" -u netta_baselines.py --parity
else
    conditions=(clean sym20 sym30 sym50 asym20 asym30 asym50 cifar100n)
    condition="${conditions[$SLURM_ARRAY_TASK_ID]}"
    "$python" -u netta_baselines.py --condition "$condition" --output "$condition"
fi
