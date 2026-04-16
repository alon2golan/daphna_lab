#!/bin/bash
#SBATCH --job-name=mh_c10_ref_log
#SBATCH --mem=24g
#SBATCH -c 4
#SBATCH --time=03:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --array=0-2%1
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/reference_validation/maxherding/cifar10/logistic/maxherding_cifar10_logistic_seed%a-%A.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/reference_validation/maxherding/cifar10/logistic/maxherding_cifar10_logistic_seed%a-%A.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/reference_validation/maxherding/cifar10/logistic"
SEED=${SLURM_ARRAY_TASK_ID}
EXP_NAME=refval_maxherding_cifar10_seed${SEED}_logistic_global

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
