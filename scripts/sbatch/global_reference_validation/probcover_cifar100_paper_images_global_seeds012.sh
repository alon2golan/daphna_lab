#!/bin/bash
#SBATCH --job-name=pc_c100_ref_img
#SBATCH --mem=32g
#SBATCH -c 8
#SBATCH --time=16:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --array=0-2%1
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/reference_validation/probcover/cifar100/paper_images/probcover_cifar100_paper_images_seed%a-%A.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/reference_validation/probcover/cifar100/paper_images/probcover_cifar100_paper_images_seed%a-%A.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/reference_validation/probcover/cifar100/paper_images"
SEED=${SLURM_ARRAY_TASK_ID}
EXP_NAME=refval_probcover_cifar100_seed${SEED}_paper_images_global

mkdir -p "$OUT_DIR"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
cd deep_al/tools

python train_al.py \
  --cfg ../configs/cifar100/al/RESNET18.yaml \
  --al prob_cover \
  --diff_method prob_cover \
  --sparse_K \
  --matrix_dtype float16 \
  --kernel_type tophat \
  --initial_delta 0.65 \
  --exp-name ${EXP_NAME} \
  --initial_size 0 \
  --budget 100 \
  --max_iter 9 \
  --seed ${SEED} \
  --eval_model_type from_images
