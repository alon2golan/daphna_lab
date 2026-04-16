#!/bin/bash
#SBATCH --job-name=mh_c100_ref_nn1
#SBATCH --mem=32g
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --array=0-2%1
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/reference_validation/maxherding/cifar100/paper_nn1/maxherding_cifar100_paper_nn1_seed%a-%A.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/reference_validation/maxherding/cifar100/paper_nn1/maxherding_cifar100_paper_nn1_seed%a-%A.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/reference_validation/maxherding/cifar100/paper_nn1"
SEED=${SLURM_ARRAY_TASK_ID}
EXP_NAME=refval_maxherding_cifar100_seed${SEED}_paper_nn1_global

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
  --budget 100 \
  --max_iter 9 \
  --seed ${SEED} \
  --eval_model_type NN1
