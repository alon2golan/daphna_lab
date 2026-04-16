#!/bin/bash
#SBATCH --job-name=mh_c10_s2_logr_ref
#SBATCH --mem=32g
#SBATCH -c 4
#SBATCH --time=08:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/sigma_probe_seed2/cifar10/logistic/global_sigma/maxherding_cifar10_seed2_logistic_global_ref-%j.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/sigma_probe_seed2/cifar10/logistic/global_sigma/maxherding_cifar10_seed2_logistic_global_ref-%j.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/maxherding/sigma_probe_seed2/cifar10/logistic/global_sigma"
SEED=${SEED_OVERRIDE:-2}
EXP_NAME=maxherding_cifar10_seed${SEED}_logistic_global_sigma

mkdir -p "$OUT_DIR"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
cd deep_al/tools

python train_al.py \
  --cfg ../configs/cifar10/al/RESNET18.yaml \
  --al max_herding \
  --diff_method max_herding \
  --kernel_type rbf \
  --initial_delta 1.0 \
  --initial_sigma 1.0 \
  --exp-name ${EXP_NAME} \
  --initial_size 0 \
  --budget 10 \
  --max_iter 9 \
  --seed ${SEED} \
  --eval_model_type logistic_from_features
