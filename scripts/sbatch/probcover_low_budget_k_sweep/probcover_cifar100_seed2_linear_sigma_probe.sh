#!/bin/bash
#SBATCH --job-name=pc_c100_s2_lin_sig
#SBATCH --mem=64g
#SBATCH -c 4
#SBATCH --time=28:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --array=0-12%4
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/probcover/low_budget_k_sweep/cifar100/linear/local_sigma/probcover_cifar100_seed2_linear_k%a-%A.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/probcover/low_budget_k_sweep/cifar100/linear/local_sigma/probcover_cifar100_seed2_linear_k%a-%A.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/probcover/low_budget_k_sweep/cifar100/linear/local_sigma"
SEED=${SEED_OVERRIDE:-2}
K_VALUES=(3 7 60 480 1440 2000 3500 5000 10000 20000 30000 35000 40000)
K=${K_VALUES[$SLURM_ARRAY_TASK_ID]}
EXP_NAME=probcover_cifar100_seed${SEED}_linear_local_sigma_k${K}

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
  --local_sigma \
  --local_sigma_k ${K} \
  --exp-name ${EXP_NAME} \
  --initial_size 0 \
  --budget 200 \
  --max_iter 14 \
  --seed ${SEED} \
  --eval_model_type from_features
