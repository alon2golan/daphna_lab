#!/usr/bin/env bash
# Portable (no-SLURM) version of run_pretrain.sbatch -- runs on the cluster OR any
# regular computer. Pretrains a solo-learn self-supervised model on CIFAR.
#
# Usage:
#   ./run_pretrain.sh simclr cifar10            # full 1000-epoch run
#   ./run_pretrain.sh byol   cifar100 100       # quick 100-epoch test
#
# Positional args:
#   1) METHOD      config name under scripts/pretrain/cifar  (simclr, byol, ...)
#   2) DATASET     cifar10 | cifar100 | stl10
#   3) MAX_EPOCHS  optional; defaults to 1000
#
# Uses the GPU automatically if torch sees one, otherwise CPU (much slower).
# wandb runs OFFLINE by default here (no account needed); set WANDB_OFFLINE=0 for live.

set -e

METHOD=${1:-simclr}
DATASET=${2:-cifar10}
MAX_EPOCHS=${3:-1000}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # .../solo-learn
ROOT=$(dirname "$SCRIPT_DIR")
cd "$SCRIPT_DIR"

[ -f "$ROOT/venv/bin/activate" ] && source "$ROOT/venv/bin/activate"

DATA_DIR="${DATA_DIR:-$ROOT/data}"
[ -d "$DATA_DIR" ] || DATA_DIR="$SCRIPT_DIR/datasets"

echo "Pretraining: method=$METHOD dataset=$DATASET epochs=$MAX_EPOCHS (data=$DATA_DIR)"

RUN_NAME="${DATASET}/${MAX_EPOCHS}ep"
FREQ=$(( MAX_EPOCHS / 10 ))
(( FREQ < 1 )) && FREQ=1

# Default to OFFLINE wandb on a laptop; export WANDB_OFFLINE=0 for live logging.
WANDB_OFFLINE_FLAG=True
[[ "${WANDB_OFFLINE:-1}" == "0" ]] && WANDB_OFFLINE_FLAG=False

python -u main_pretrain.py \
  --config-path scripts/pretrain/cifar \
  --config-name "${METHOD}.yaml" \
  data.dataset="${DATASET}" \
  data.train_path="${DATA_DIR}" \
  data.val_path="${DATA_DIR}" \
  name="${METHOD}-${DATASET}-${MAX_EPOCHS}ep" \
  ++checkpoint.run_name="${RUN_NAME}" \
  ++checkpoint.frequency="${FREQ}" \
  ++checkpoint.keep_prev=True \
  ++knn_eval.enabled=True \
  ++enable_progress_bar=False \
  max_epochs="${MAX_EPOCHS}" \
  wandb.enabled=True \
  ++wandb.offline="${WANDB_OFFLINE_FLAG}"
