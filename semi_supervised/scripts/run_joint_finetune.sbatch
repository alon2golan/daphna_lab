#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=32g
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/cs/labs/daphna/%u/projects/daphna_lab/semi_supervised/slurm_logs/%x_%j.out
#SBATCH --error=/cs/labs/daphna/%u/projects/daphna_lab/semi_supervised/slurm_logs/%x_%j.err
#SBATCH --killable
#SBATCH --requeue

# SLURM runner for joint SimCLR+SupCon fine-tuning: baseline (the frozen 1000ep
# encoder) + one joint run per --unsup_weights value. Logs to its OWN wandb
# project (semi-supervised-joint), separate from supcon_finetune.
#
# NOTE: default --unsup_source unlabeled runs the SimCLR term over the FULL pool,
# so each epoch is a full pass over ~50k images and the sweep is several such
# trainings -- hence the 24h wall-time. Trim EPOCHS / UNSUP_WEIGHTS for a quick
# first look, or raise --time for a big sweep.
#
# Usage:
#   sbatch --job-name=joint_simclr_cifar100_L10 \
#          run_joint_finetune.sbatch trained_models/simclr/cifar100/1000ep cifar100 10
#
# Positional: 1) BASE_CKPT (rel to solo-learn or abs) 2) DATASET 3) L
# Env: EPOCHS (50), UNSUP_WEIGHTS (1,2,5,10), SEED (1), WANDB_OFFLINE=1, EXTRA="..."

set -e

BASE_CKPT=${1:?usage: sbatch run_joint_finetune.sbatch BASE_CKPT DATASET L}
DATASET=${2:?usage: sbatch run_joint_finetune.sbatch BASE_CKPT DATASET L}
L=${3:?usage: sbatch run_joint_finetune.sbatch BASE_CKPT DATASET L}

case "$DATASET" in cifar10) NCLASSES=10 ;; cifar100) NCLASSES=100 ;; *) NCLASSES=100 ;; esac
N_LABELED=$(( L * NCLASSES ))
EPOCHS=${EPOCHS:-50}
UNSUP_WEIGHTS=${UNSUP_WEIGHTS:-1,2,5,10}
SEED=${SEED:-1}

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
DATA_DIR="$ROOT/data"
RESULTS_ROOT="$ROOT/semi_supervised/results"
case "$BASE_CKPT" in /*) ;; *) BASE_CKPT="$ROOT/solo-learn/$BASE_CKPT" ;; esac

cd "$ROOT"
source "$ROOT/venv/bin/activate"
mkdir -p "$ROOT/semi_supervised/slurm_logs"

WANDB_OFFLINE_FLAG=""
[[ "${WANDB_OFFLINE:-0}" == "1" ]] && WANDB_OFFLINE_FLAG="--wandb_offline"

echo "Joint FT: base=$BASE_CKPT dataset=$DATASET L=$L n_labeled=$N_LABELED epochs=$EPOCHS weights=$UNSUP_WEIGHTS EXTRA=${EXTRA:-}"

python -u -m semi_supervised.algorithms.joint_finetune \
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
