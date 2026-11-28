#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=npc-graph-weights
set -euo pipefail
projects=/cs/labs/daphna/tomn/projects
work=$1
graph_root=$2
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$graph_root/source:$projects/cifar_matched_npc_20261127/source"
"$projects/daphna_lab/.venv-vast/bin/python" -u "$work/source/rank_weights.py" score \
    "$projects/cifar_matched_npc_20261127" "$graph_root/results" "$SLURM_ARRAY_TASK_ID" "$work/results"
