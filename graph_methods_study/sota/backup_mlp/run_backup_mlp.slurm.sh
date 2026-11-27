#!/bin/bash
#SBATCH --partition=short
#SBATCH --gres=gg:g0:1,vmem:8g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
set -euo pipefail
cd "$1"
p="$(dirname "$PWD")"
herding="$p/cifar100_herding_20261119"
export PYTHONPATH="$herding/source:$p/vast_followup_20261024/runtime_deps:$p/vast_expanded_20261019/runtime_deps"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
args=("$PWD" "$herding/study" "$p/cifar100_js_followup_20261120/study" "$p/cifar_exact_backup_20261126/orders.json")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then args+=(--index "$SLURM_ARRAY_TASK_ID"); fi
"$p/daphna_lab/.venv-vast/bin/python" -u backup_mlp.py "${args[@]}"
