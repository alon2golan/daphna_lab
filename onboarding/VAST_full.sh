#!/bin/bash
#SBATCH --mem=40g
#SBATCH -c 8
#SBATCH --time=96:00:00
#SBATCH --gres=gg:g0:1,vmem:20g
#SBATCH --output=vast_full_%j.log
#SBATCH --error=vast_full_%j.log
set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-/cs/labs/daphna/$USER/projects/daphna_lab}"
VAST_PYTHON="${VAST_PYTHON:-$REPO_DIR/.venv-vast/bin/python}"
VAST_DEVICE="${VAST_DEVICE:-cuda}"
SEED="${SEED:-0}"
# Create these three manifests using deep_al/vast/manifest.example.yaml.
VAST_MANIFEST_DIR="${VAST_MANIFEST_DIR:-$REPO_DIR/data/vast}"
VAST_OUTPUT_DIR="${VAST_OUTPUT_DIR:-$REPO_DIR/results/vast/full_seed${SEED}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export VAST_FAISS_THREADS="${VAST_FAISS_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
cd "$REPO_DIR"
extra=()
if [[ "${VAST_DRY_RUN:-0}" == 1 ]]; then extra+=(--dry-run); fi
echo "VAST full | host=$(hostname) | seed=$SEED | device=$VAST_DEVICE | output=$VAST_OUTPUT_DIR"
echo "Experimental literal-paper propagation: numerical reproduction remains unconfirmed."
"$VAST_PYTHON" -u deep_al/tools/run_vast_grid.py \
  --manifests "$VAST_MANIFEST_DIR/cifar100.yaml" \
              "$VAST_MANIFEST_DIR/imagenet100.yaml" \
              "$VAST_MANIFEST_DIR/tinyimagenet.yaml" \
  --seed "$SEED" --epochs 300 --device "$VAST_DEVICE" \
  --propagation literal-paper --output-dir "$VAST_OUTPUT_DIR" "${extra[@]}"
