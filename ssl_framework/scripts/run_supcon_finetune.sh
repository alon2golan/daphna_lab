#!/usr/bin/env bash
# Portable (no-SLURM) single-case runner: baseline + supcon_only + supcon_prop
# (all kernels) for ONE (embedding, dataset, budget), end to end on this box's GPU.
# Mirrors run_supcon_finetune.sbatch but runs --mode all in the foreground.
#
# Usage:
#   ./run_supcon_finetune.sh trained_models/simclr/cifar100/1000ep cifar100 10
#   ./run_supcon_finetune.sh trained_models/byol/cifar10/1000ep    cifar10  25
#
# Positional args:
#   1) BASE_CKPT    checkpoint dir (relative to solo-learn) or absolute
#   2) DATASET      cifar10 | cifar100
#   3) L            per-class budget (n_labeled = L * num_classes)
#
# Extra flags via EXTRA, e.g. EXTRA="--sim_threshold 0.1 --conf_threshold 0.8".
# wandb offline: WANDB_OFFLINE=1 ./run_supcon_finetune.sh ...

set -e

BASE_CKPT=${1:?usage: ./run_supcon_finetune.sh BASE_CKPT DATASET L}
DATASET=${2:?usage: ./run_supcon_finetune.sh BASE_CKPT DATASET L}
L=${3:?usage: ./run_supcon_finetune.sh BASE_CKPT DATASET L}

case "$DATASET" in cifar10) NCLASSES=10 ;; cifar100) NCLASSES=100 ;; *) NCLASSES=100 ;; esac
N_LABELED=$(( L * NCLASSES ))

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # .../ssl_framework/scripts
ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")
# a relative checkpoint is taken under solo-learn (where trained_models lives).
case "$BASE_CKPT" in /*) ;; *) BASE_CKPT="$ROOT/solo-learn/$BASE_CKPT" ;; esac

cd "$ROOT"
[ -f "$ROOT/venv/bin/activate" ] && source "$ROOT/venv/bin/activate"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
RESULTS_ROOT="$ROOT/ssl_framework/results"

WANDB_OFFLINE_FLAG=""
[[ "${WANDB_OFFLINE:-0}" == "1" ]] && WANDB_OFFLINE_FLAG="--wandb_offline"

echo "SupCon FT (mode all): base=$BASE_CKPT dataset=$DATASET L=$L n_labeled=$N_LABELED"

python -u -m ssl_framework.old.supcon_finetune \
  --dataset "$DATASET" \
  --base_checkpoint "$BASE_CKPT" \
  --data_dir "$DATA_DIR" \
  --results_root "$RESULTS_ROOT" \
  --mode all \
  --n_labeled "$N_LABELED" \
  --per_class_budget "$L" \
  --with_propagation \
  --kernels gaussian,laplacian,cosine \
  --auto_thresholds 2 \
  $WANDB_OFFLINE_FLAG \
  ${EXTRA:-}
