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
#   sbatch run_extract.sbatch trained_models/simclr/cifar100/1000ep cifar100
#   sbatch run_extract.sbatch trained_models/simclr/cifar100/1000ep cifar100 embeddings/my_run
#
# Positional args:
#   1) CKPT_DIR     checkpoint dir with the .ckpt and args.json (from pretraining)
#   2) DATASET      cifar10 | cifar100 | stl10
#   3) OUTPUT_DIR   optional; defaults to embeddings/<method>/<dataset>/<epochs>ep,
#                   mirroring the checkpoint layout
#                   (e.g. embeddings/simclr/cifar100/1000ep)
#
# Produces train_features.npy / test_features.npy (+ *_labels.npy) under OUTPUT_DIR.

set -e

CKPT_DIR=${1:?usage: sbatch run_extract.sbatch CKPT_DIR DATASET [OUTPUT_DIR]}
DATASET=${2:?usage: sbatch run_extract.sbatch CKPT_DIR DATASET [OUTPUT_DIR]}
# Mirror trained_models/<method>/<dataset>/<epochs>ep as
# embeddings/<method>/<dataset>/<epochs>ep so the extracted features stay
# traceable to the exact checkpoint they came from.
CKPT_DIR=${CKPT_DIR%/}
EPOCHS_NAME=$(basename "$CKPT_DIR")                              # e.g. 1000ep
DATASET_NAME=$(basename "$(dirname "$CKPT_DIR")")               # e.g. cifar100
METHOD_NAME=$(basename "$(dirname "$(dirname "$CKPT_DIR")")")   # e.g. simclr
OUTPUT_DIR=${3:-embeddings/$METHOD_NAME/$DATASET_NAME/$EPOCHS_NAME}

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
