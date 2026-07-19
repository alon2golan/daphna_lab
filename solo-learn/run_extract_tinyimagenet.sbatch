#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=16g
#SBATCH -c 4
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:l4:1
#SBATCH --output=slurm_logs/extract_tin_%j.out
#SBATCH --error=slurm_logs/extract_tin_%j.err
#SBATCH --killable
#SBATCH --requeue

# Extract backbone embeddings from a tiny-imagenet checkpoint into .npy files.
# Mirrors run_extract.sbatch but feeds the two class-folder trees (train +
# val_sorted) and the tinyimagenet transform (64x64, no resize).
#
# Usage:
#   sbatch run_extract_tinyimagenet.sbatch trained_models/simclr/tinyimagenet/1000ep
#   sbatch run_extract_tinyimagenet.sbatch trained_models/byol/tinyimagenet/1000ep embeddings/my_run
#
# Positional args:
#   1) CKPT_DIR     checkpoint dir with the .ckpt and args.json
#   2) OUTPUT_DIR   optional; defaults to embeddings/<method>/tinyimagenet/<epochs>ep
#
# Produces train_features.npy / test_features.npy (+ *_normalized, *_labels).

set -e

CKPT_DIR=${1:?usage: sbatch run_extract_tinyimagenet.sbatch CKPT_DIR [OUTPUT_DIR]}
CKPT_DIR=${CKPT_DIR%/}
# Mirror the checkpoint layout under embeddings/ so features stay traceable.
EPOCHS_NAME=$(basename "$CKPT_DIR")                              # e.g. 1000ep
DATASET_NAME=$(basename "$(dirname "$CKPT_DIR")")               # tinyimagenet
METHOD_NAME=$(basename "$(dirname "$(dirname "$CKPT_DIR")")")   # simclr | byol
OUTPUT_DIR=${2:-embeddings/$METHOD_NAME/$DATASET_NAME/$EPOCHS_NAME}

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
TRAIN_DIR="$ROOT/data/tiny-imagenet-200/train"
VAL_DIR="$ROOT/data/tiny-imagenet-200/val_sorted"
cd "$ROOT"
source "$ROOT/venv/bin/activate"
cd "$ROOT/solo-learn"
mkdir -p slurm_logs

echo "Extracting: ckpt=$CKPT_DIR -> $OUTPUT_DIR"
echo "  train=$TRAIN_DIR  val=$VAL_DIR"

# extract_embeddings.py already accepts separate train/val paths; tinyimagenet is
# now a first-class dataset so it gets the right (64px, no-resize) T_val.
python -u extract_embeddings.py \
  --checkpoint_dir "$CKPT_DIR" \
  --dataset tinyimagenet \
  --train_data_path "$TRAIN_DIR" \
  --val_data_path "$VAL_DIR" \
  --output_dir "$OUTPUT_DIR"
