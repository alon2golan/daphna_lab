#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=npc-rank-confirm
set -euo pipefail

work=$1
projects=/cs/labs/daphna/tomn/projects
official="$projects/cifar_matched_npc_20261127"
export PYTHONPATH="$official/source:$official/runtime_deps:$projects/vast_followup_20261024/runtime_deps:$projects/vast_expanded_20261019/runtime_deps"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="$work/matplotlib_cache"
"$projects/daphna_lab/.venv-vast/bin/python" -u "$work/source/run_fresh.py" \
    "$SLURM_ARRAY_TASK_ID" "$work" "$official"
