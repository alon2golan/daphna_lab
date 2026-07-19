#!/usr/bin/env bash
# Joint SimCLR+SupCon fine-tuning sweep over the unsupervised weight, on top of an
# existing pretrained SimCLR checkpoint. Baseline = the frozen 1000ep encoder.
# Logs to its OWN wandb project (semi-supervised-joint) so it never mixes with the
# old supcon_finetune runs.
#
# Usage:
#   ./run_joint_finetune.sh trained_models/simclr/cifar100/1000ep cifar100 10
# Positional: 1) BASE_CKPT (rel to solo-learn or absolute) 2) DATASET 3) L
# Env: EPOCHS (default 50), UNSUP_WEIGHTS (default 1,2,5,10), SEED (1),
#      WANDB_OFFLINE=1, EXTRA="...".

set -e

BASE_CKPT=${1:?usage: ./run_joint_finetune.sh BASE_CKPT DATASET L}
DATASET=${2:?usage: ./run_joint_finetune.sh BASE_CKPT DATASET L}
L=${3:?usage: ./run_joint_finetune.sh BASE_CKPT DATASET L}

case "$DATASET" in cifar10) NCLASSES=10 ;; cifar100) NCLASSES=100 ;; *) NCLASSES=100 ;; esac
N_LABELED=$(( L * NCLASSES ))
EPOCHS=${EPOCHS:-50}
UNSUP_WEIGHTS=${UNSUP_WEIGHTS:-1,2,5,10}
SEED=${SEED:-1}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # .../ssl_framework/scripts
ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")
case "$BASE_CKPT" in /*) ;; *) BASE_CKPT="$ROOT/solo-learn/$BASE_CKPT" ;; esac

cd "$ROOT"
[ -f "$ROOT/venv/bin/activate" ] && source "$ROOT/venv/bin/activate"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
RESULTS_ROOT="$ROOT/ssl_framework/results"

WANDB_OFFLINE_FLAG=""
[[ "${WANDB_OFFLINE:-0}" == "1" ]] && WANDB_OFFLINE_FLAG="--wandb_offline"

echo "Joint FT: base=$BASE_CKPT dataset=$DATASET L=$L n_labeled=$N_LABELED epochs=$EPOCHS weights=$UNSUP_WEIGHTS"

python -u -m ssl_framework.algorithms.joint_finetune \
  --dataset "$DATASET" \
  --base_checkpoint "$BASE_CKPT" \
  --data_dir "$DATA_DIR" \
  --results_root "$RESULTS_ROOT" \
  --n_labeled "$N_LABELED" \
  --per_class_budget "$L" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --unsup_weights "$UNSUP_WEIGHTS" \
  $WANDB_OFFLINE_FLAG \
  ${EXTRA:-}
