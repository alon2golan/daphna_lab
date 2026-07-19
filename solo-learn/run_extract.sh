#!/usr/bin/env bash
# Portable (no-SLURM) version of run_extract.sbatch -- runs on the cluster OR any
# regular computer. Extracts backbone embeddings from one checkpoint into .npy.
#
# Usage:
#   ./run_extract.sh trained_models/simclr/cifar100/1000ep cifar100
#   ./run_extract.sh trained_models/simclr/cifar100/1000ep cifar100 embeddings/my_run
#
# Positional args:
#   1) CKPT_DIR     checkpoint dir with the .ckpt and args.json
#   2) DATASET      cifar10 | cifar100 | stl10
#   3) OUTPUT_DIR   optional; defaults to embeddings/<method>/<dataset>/<epochs>ep
#
# Produces train_features.npy / test_features.npy (+ *_labels.npy) under OUTPUT_DIR.

set -e

CKPT_DIR=${1:?usage: ./run_extract.sh CKPT_DIR DATASET [OUTPUT_DIR]}
DATASET=${2:?usage: ./run_extract.sh CKPT_DIR DATASET [OUTPUT_DIR]}

CKPT_DIR=${CKPT_DIR%/}
EPOCHS_NAME=$(basename "$CKPT_DIR")
DATASET_NAME=$(basename "$(dirname "$CKPT_DIR")")
METHOD_NAME=$(basename "$(dirname "$(dirname "$CKPT_DIR")")")
OUTPUT_DIR=${3:-embeddings/$METHOD_NAME/$DATASET_NAME/$EPOCHS_NAME}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # .../solo-learn
ROOT=$(dirname "$SCRIPT_DIR")
cd "$SCRIPT_DIR"

[ -f "$ROOT/venv/bin/activate" ] && source "$ROOT/venv/bin/activate"

DATA_DIR="${DATA_DIR:-$ROOT/data}"
[ -d "$DATA_DIR" ] || DATA_DIR="$SCRIPT_DIR/datasets"

echo "Extracting: ckpt=$CKPT_DIR dataset=$DATASET -> $OUTPUT_DIR (data=$DATA_DIR)"

python -u extract_embeddings.py \
  --checkpoint_dir "$CKPT_DIR" \
  --dataset "$DATASET" \
  --train_data_path "$DATA_DIR" \
  --val_data_path "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR"
