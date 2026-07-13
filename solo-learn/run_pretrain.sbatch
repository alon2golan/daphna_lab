#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=32g
#SBATCH -c 8
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:l4:1
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
# The run auto-resumes from its latest checkpoint if requeued (auto_resume is
# enabled in the config), so a 48h wall clock split across preemptions is fine.
# Weights & Biases logging is disabled here; drop wandb.enabled=False to turn it
# back on (set your entity in scripts/pretrain/cifar/wandb/private.yaml first).

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

python -u main_pretrain.py \
  --config-path scripts/pretrain/cifar \
  --config-name "${METHOD}.yaml" \
  data.dataset="${DATASET}" \
  data.train_path="${DATA_DIR}" \
  data.val_path="${DATA_DIR}" \
  name="${METHOD}-${DATASET}" \
  max_epochs="${MAX_EPOCHS}" \
  wandb.enabled=False
