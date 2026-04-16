#!/bin/bash
#SBATCH --job-name=pc_c100_s2_img_ref
#SBATCH --mem=24g
#SBATCH -c 4
#SBATCH --time=24:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/probcover/low_budget_k_sweep/cifar100/images/global_sigma/probcover_cifar100_seed2_images_global_ref-%j.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/probcover/low_budget_k_sweep/cifar100/images/global_sigma/probcover_cifar100_seed2_images_global_ref-%j.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/probcover/low_budget_k_sweep/cifar100/images/global_sigma"
SEED=${SEED_OVERRIDE:-2}
EXP_NAME=probcover_cifar100_seed${SEED}_images_global_sigma

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
  --no_features_normalized \
  --exp-name ${EXP_NAME} \
  --initial_size 0 \
  --budget 200 \
  --max_iter 14 \
  --seed ${SEED} \
  --eval_model_type from_images
