#!/bin/bash
#SBATCH --job-name=mh_c100_ref
#SBATCH --mem=32g
#SBATCH -c 4
#SBATCH --time=06:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/low_budget_k_sweep/cifar100/global_sigma/reference/maxherding_cifar100_global_ref_seed0-%j.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/low_budget_k_sweep/cifar100/global_sigma/reference/maxherding_cifar100_global_ref_seed0-%j.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/maxherding/low_budget_k_sweep/cifar100/global_sigma/reference"
SEED=${SEED_OVERRIDE:-0}
EXP_NAME=low_budget_maxherding_cifar100_global_sigma_seed${SEED}

mkdir -p "$OUT_DIR"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
cd deep_al/tools

python train_al.py \
  --cfg ../configs/cifar100/al/RESNET18.yaml \
  --al max_herding \
  --diff_method max_herding \
  --kernel_type rbf \
  --initial_delta 1.0 \
  --initial_sigma 1.0 \
  --exp-name ${EXP_NAME} \
  --initial_size 0 \
  --budget 200 \
  --max_iter 14 \
  --seed ${SEED} \
  --eval_model_type NN1
