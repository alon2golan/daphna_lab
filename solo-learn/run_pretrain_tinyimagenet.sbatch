#!/bin/bash
#SBATCH -p long
#SBATCH --mem=64g
#SBATCH -c 16
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --output=slurm_logs/pretrain_tin_%j.out
#SBATCH --error=slurm_logs/pretrain_tin_%j.err
#SBATCH --killable
#SBATCH --requeue

# Pretrain a solo-learn SSL model (simclr / byol) on tiny-imagenet (64x64, 200
# classes). Mirrors run_pretrain.sbatch but points at the tinyimagenet config
# (crop_size 64, cifar-style small resnet stem) and the class-folder data tree.
#
# Prereq (run ONCE, cheap): build the ImageFolder-compatible val split:
#   python prepare_tinyimagenet_val.py --root ../data/tiny-imagenet-200
#
# Usage:
#   sbatch run_pretrain_tinyimagenet.sbatch simclr           # full 1000-epoch run
#   sbatch run_pretrain_tinyimagenet.sbatch byol   100       # quick 100-epoch test
#
# Positional args:
#   1) METHOD      config name under scripts/pretrain/tinyimagenet (simclr, byol)
#   2) MAX_EPOCHS  optional; defaults to 1000
#
# Checkpoints -> trained_models/<method>/tinyimagenet/<epochs>ep/ (ckpts + args.json).
# Same auto-resume / offline-wandb behavior as the cifar runner.

set -e

METHOD=${1:-simclr}
MAX_EPOCHS=${2:-1000}
DATASET=tinyimagenet

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
# tiny-imagenet lives under the repo data dir as class-folder trees:
#   train/<wnid>/images/*.JPEG   and   val_sorted/<wnid>/*.JPEG
TRAIN_DIR="$ROOT/data/tiny-imagenet-200/train"
VAL_DIR="$ROOT/data/tiny-imagenet-200/val_sorted"
cd "$ROOT"
source "$ROOT/venv/bin/activate"
cd "$ROOT/solo-learn"
mkdir -p slurm_logs

echo "Pretraining: method=$METHOD dataset=$DATASET epochs=$MAX_EPOCHS"
echo "  train=$TRAIN_DIR  val=$VAL_DIR"

# Same self-documenting layout as cifar: trained_models/<method>/tinyimagenet/<epochs>ep
RUN_NAME="${DATASET}/${MAX_EPOCHS}ep"

# ~10 evenly-spaced snapshots plus epoch 0.
FREQ=$(( MAX_EPOCHS / 10 ))
(( FREQ < 1 )) && FREQ=1

# wandb online by default; sbatch --export=ALL,WANDB_OFFLINE=1 ... for offline.
WANDB_OFFLINE_FLAG=False
[[ "${WANDB_OFFLINE:-0}" == "1" ]] && WANDB_OFFLINE_FLAG=True

python -u main_pretrain.py \
  --config-path scripts/pretrain/tinyimagenet \
  --config-name "${METHOD}.yaml" \
  data.dataset="${DATASET}" \
  data.train_path="${TRAIN_DIR}" \
  data.val_path="${VAL_DIR}" \
  name="${METHOD}-${DATASET}-${MAX_EPOCHS}ep" \
  ++checkpoint.run_name="${RUN_NAME}" \
  ++checkpoint.frequency="${FREQ}" \
  ++checkpoint.keep_prev=True \
  ++knn_eval.enabled=True \
  ++enable_progress_bar=False \
  max_epochs="${MAX_EPOCHS}" \
  wandb.enabled=True \
  ++wandb.offline="${WANDB_OFFLINE_FLAG}"
