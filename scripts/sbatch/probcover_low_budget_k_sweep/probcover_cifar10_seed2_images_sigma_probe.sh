#!/bin/bash
#SBATCH --job-name=pc_c10_s2_img_sig
#SBATCH --mem=64g
#SBATCH -c 4
#SBATCH --time=28:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --array=0-12%4
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/probcover/low_budget_k_sweep/cifar10/images/local_sigma/probcover_cifar10_seed2_images_k%a-%A.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/probcover/low_budget_k_sweep/cifar10/images/local_sigma/probcover_cifar10_seed2_images_k%a-%A.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/probcover/low_budget_k_sweep/cifar10/images/local_sigma"
SEED=${SEED_OVERRIDE:-2}
K_VALUES=(3 7 60 480 1440 2000 3500 5000 10000 20000 30000 35000 40000)
K=${K_VALUES[$SLURM_ARRAY_TASK_ID]}
EXP_NAME=probcover_cifar10_seed${SEED}_images_local_sigma_k${K}

mkdir -p "$OUT_DIR"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
cd deep_al/tools

python train_al.py \
  --cfg ../configs/cifar10/al/RESNET18.yaml \
  --al prob_cover \
  --diff_method prob_cover \
  --sparse_K \
  --matrix_dtype float16 \
  --kernel_type tophat \
  --initial_delta 0.75 \
  --no_features_normalized \
  --local_sigma \
  --local_sigma_k ${K} \
  --exp-name ${EXP_NAME} \
  --initial_size 0 \
  --budget 10 \
  --max_iter 9 \
  --seed ${SEED} \
  --eval_model_type from_images
