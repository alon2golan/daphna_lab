#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4
set -euo pipefail
cd "$1"
p="$(dirname "$PWD")"
netta="$p/cifar_netta_baselines_20261127"
export PYTHONPATH="$netta:$p/vast_followup_20261024/source:$p/cifar_active_audit_20261126/human:$p/vast_followup_20261024/runtime_deps:$p/vast_expanded_20261019/runtime_deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=2 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
"$p/daphna_lab/.venv-vast/bin/python" -u conditional_votes.py "$SLURM_ARRAY_TASK_ID" "$netta" "case_${SLURM_ARRAY_TASK_ID}.json"
