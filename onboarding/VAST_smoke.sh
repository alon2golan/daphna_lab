#!/bin/bash
#SBATCH --mem=8g
#SBATCH -c 4
#SBATCH --time=00:20:00
#SBATCH --gres=gg:g0:1,vmem:8g
#SBATCH --output=vast_smoke_%j.log
#SBATCH --error=vast_smoke_%j.log
set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-/cs/labs/daphna/$USER/projects/daphna_lab}"
VAST_PYTHON="${VAST_PYTHON:-$REPO_DIR/.venv-vast/bin/python}"
VAST_DEVICE="${VAST_DEVICE:-cuda}"
SEED="${SEED:-0}"
VAST_OUTPUT_DIR="${VAST_OUTPUT_DIR:-$REPO_DIR/results/vast/smoke_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export VAST_FAISS_THREADS="${VAST_FAISS_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
cd "$REPO_DIR"
echo "VAST smoke | host=$(hostname) | seed=$SEED | device=$VAST_DEVICE | output=$VAST_OUTPUT_DIR"
"$VAST_PYTHON" -u deep_al/tools/smoke_vast.py \
  --seed "$SEED" --device "$VAST_DEVICE" --output-dir "$VAST_OUTPUT_DIR"
echo "Smoke passed. Synthetic data checks execution, not real-data alignment or paper accuracy."
