#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=32g
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/cs/labs/daphna/%u/projects/daphna_lab/ssl_framework/slurm_logs/%x_%j.out
#SBATCH --error=/cs/labs/daphna/%u/projects/daphna_lab/ssl_framework/slurm_logs/%x_%j.err
#SBATCH --killable
#SBATCH --requeue

# Generic single-job runner for one SupCon-fine-tuning invocation. Everything
# past the base encoder + dataset is passed through EXTRA, so this one file backs
# both the baseline jobs and the per-budget jobs the sweep launcher submits.
# Logs land in ssl_framework/slurm_logs/<jobname>_<jobid>.{out,err} (absolute
# path above), so it does not matter where you sbatch from.
#
# Usage (normally called by run_supcon_sweep_all.sh, but works standalone):
#   EXTRA="--mode baseline" \
#     sbatch --job-name=scft_simclr_cifar100_baseline \
#            run_supcon_finetune.sbatch trained_models/simclr/cifar100/1000ep cifar100
#   EXTRA="--mode budget --n_labeled 1000 --per_class_budget 10 --with_propagation \
#          --kernels gaussian,laplacian,cosine --bandwidth auto" \
#     sbatch --job-name=scft_simclr_cifar100_L10 \
#            run_supcon_finetune.sbatch trained_models/simclr/cifar100/1000ep cifar100
#
# Positional args:
#   1) BASE_CKPT   checkpoint dir (relative to solo-learn, like run_extract) or abs
#   2) DATASET     cifar10 | cifar100
# Env:
#   EXTRA          all remaining flags (mode, n_labeled, budget, kernels, ...)
#   WANDB_OFFLINE  =1 to log offline

set -e

BASE_CKPT=${1:?usage: sbatch run_supcon_finetune.sbatch BASE_CKPT DATASET   (rest via EXTRA)}
DATASET=${2:?usage: sbatch run_supcon_finetune.sbatch BASE_CKPT DATASET   (rest via EXTRA)}

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
DATA_DIR="$ROOT/data"
case "$BASE_CKPT" in /*) ;; *) BASE_CKPT="$ROOT/solo-learn/$BASE_CKPT" ;; esac

cd "$ROOT"
source "$ROOT/venv/bin/activate"
mkdir -p "$ROOT/ssl_framework/slurm_logs"

WANDB_OFFLINE_FLAG=""
[[ "${WANDB_OFFLINE:-0}" == "1" ]] && WANDB_OFFLINE_FLAG="--wandb_offline"

echo "SupCon FT: base=$BASE_CKPT dataset=$DATASET (data=$DATA_DIR) EXTRA=${EXTRA:-}"

python -u -m ssl_framework.old.supcon_finetune \
  --dataset "$DATASET" \
  --base_checkpoint "$BASE_CKPT" \
  --data_dir "$DATA_DIR" \
  $WANDB_OFFLINE_FLAG \
  ${EXTRA:-}
