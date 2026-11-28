#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --job-name=npc-cap-replay
set -euo pipefail

phase=$1
work=$2
official=$3
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
projects=/cs/labs/daphna/tomn/projects
python="$projects/daphna_lab/.venv-vast/bin/python"
if [[ "$phase" = prepare ]]; then
    "$python" -u "$work/source/replay.py" prepare "$official" "$work/results"
else
    "$python" -u "$work/source/replay.py" score "$official" "$work/results" --task "$SLURM_ARRAY_TASK_ID"
fi
