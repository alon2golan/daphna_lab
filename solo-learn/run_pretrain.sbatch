#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=32g
#SBATCH -c 8
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm_logs/pretrain_%j.out
#SBATCH --error=slurm_logs/pretrain_%j.err
#SBATCH --killable
#SBATCH --requeue

# Pretrain a solo-learn self-supervised model (SimCLR / BYOL / ...) on CIFAR.
#
# Usage:
#   sbatch run_pretrain.sbatch simclr cifar10            # full 1000-epoch run
#   sbatch run_pretrain.sbatch byol   cifar100 100       # quick 100-epoch test
#
# Positional args:
#   1) METHOD      config name under scripts/pretrain/cifar  (simclr, byol, ...)
#   2) DATASET     cifar10 | cifar100 | stl10
#   3) MAX_EPOCHS  optional; defaults to 1000 (the paper-reproduction length)
#
# Checkpoints are written to a self-documenting folder:
#   trained_models/<method>/<dataset>/<epochs>ep/
#     e.g. trained_models/simclr/cifar100/1000ep/
#          <method>-<dataset>-<epochs>ep-ep=<N>.ckpt  +  args.json
#   ("<epochs>ep" = number of training epochs, not AL episodes.)
#
# The run auto-resumes from its latest checkpoint if requeued (auto_resume is
# enabled in the config), so a 48h wall clock split across preemptions is fine.
# Weights & Biases logging runs in OFFLINE mode (no account/internet needed);
# metrics land under solo-learn/wandb/offline-run-*/ -- "wandb sync <dir>" to view.
# Online (live) logging: set wandb.offline=False and an entity in
# scripts/pretrain/cifar/wandb/private.yaml.
# Also on: knn_eval (val_knn_acc1) and 10 evenly-spaced checkpoints (+ epoch 0,
# toggle with ++checkpoint.save_initial=False).

set -e

METHOD=${1:-simclr}
DATASET=${2:-cifar10}
MAX_EPOCHS=${3:-1000}

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
# Reuse the CIFAR copies already extracted under the repo instead of
# re-downloading into solo-learn/datasets. torchvision finds cifar-10-batches-py
# / cifar-100-python under this root and skips the download.
DATA_DIR="$ROOT/data"
cd "$ROOT"
source "$ROOT/venv/bin/activate"
cd "$ROOT/solo-learn"
mkdir -p slurm_logs

echo "Pretraining: method=$METHOD dataset=$DATASET epochs=$MAX_EPOCHS (data=$DATA_DIR)"

# Self-documenting run identity:
#   RUN_NAME  -> checkpoint subpath under the method dir:
#               trained_models/<method>/<dataset>/<epochs>ep
#   name      -> ckpt filename + wandb display: <method>-<dataset>-<epochs>ep
# (method is already the parent dir; dataset is a folder, epochs its subfolder.)
RUN_NAME="${DATASET}/${MAX_EPOCHS}ep"

# Checkpoint every 10% of training (so ~10 evenly-spaced snapshots, plus epoch 0),
# derived from the epoch count instead of a hardcoded number. Never below 1.
FREQ=$(( MAX_EPOCHS / 10 ))
(( FREQ < 1 )) && FREQ=1

# wandb online (live) by default; for offline logging:
#   sbatch --export=ALL,WANDB_OFFLINE=1 run_pretrain.sbatch simclr cifar100
WANDB_OFFLINE_FLAG=False
[[ "${WANDB_OFFLINE:-0}" == "1" ]] && WANDB_OFFLINE_FLAG=True

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
