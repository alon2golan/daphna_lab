#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=00:25:00
#SBATCH --job-name=npc-rank-score
set -euo pipefail

work=$1
shift
projects=/cs/labs/daphna/tomn/projects
official="$projects/cifar_matched_npc_20261127"
geometry="$projects/cifar_npc_cap_exact_20261128"
weights="$projects/cifar_npc_graph_weights_20261128"
export PYTHONPATH="$weights/source:$geometry/source:$official/source"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
"$projects/daphna_lab/.venv-vast/bin/python" -u "$work/source/score.py" \
    "$work" "$official" "$geometry/results/geometry.npz" "${SLURM_ARRAY_TASK_ID:-0}" "$@"
