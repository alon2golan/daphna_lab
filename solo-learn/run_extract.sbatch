#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=16g
#SBATCH -c 4
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:l4:1
#SBATCH --output=slurm_logs/extract_%j.out
#SBATCH --error=slurm_logs/extract_%j.err
#SBATCH --killable
#SBATCH --requeue

# Extract backbone embeddings from a solo-learn checkpoint into .npy files.
#
# Usage:
#   sbatch run_extract.sbatch trained_models/simclr/0 cifar100
#   sbatch run_extract.sbatch trained_models/simclr/0 cifar100 embeddings/my_run
#
# Positional args:
#   1) CKPT_DIR     checkpoint dir with the .ckpt and args.json (from pretraining)
#   2) DATASET      cifar10 | cifar100 | stl10
#   3) OUTPUT_DIR   optional; defaults to embeddings/<basename-of-CKPT_DIR>
#
# Produces train_features.npy / test_features.npy (+ *_labels.npy) under OUTPUT_DIR.

set -e

CKPT_DIR=${1:?usage: sbatch run_extract.sbatch CKPT_DIR DATASET [OUTPUT_DIR]}
DATASET=${2:?usage: sbatch run_extract.sbatch CKPT_DIR DATASET [OUTPUT_DIR]}
OUTPUT_DIR=${3:-embeddings/$(basename "$CKPT_DIR")}

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
DATA_DIR="$ROOT/data"
cd "$ROOT"
source "$ROOT/venv/bin/activate"
cd "$ROOT/solo-learn"
mkdir -p slurm_logs

echo "Extracting: ckpt=$CKPT_DIR dataset=$DATASET -> $OUTPUT_DIR (data=$DATA_DIR)"

python -u extract_embeddings.py \
  --checkpoint_dir "$CKPT_DIR" \
  --dataset "$DATASET" \
  --train_data_path "$DATA_DIR" \
  --val_data_path "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR"
