#!/bin/bash
#SBATCH --job-name=mh_c10_cut_sig
#SBATCH --mem=64g
#SBATCH -c 4
#SBATCH --time=24:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --array=0-23%1
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/cutoff_seeds345/cifar10/nn1/local_sigma/maxherding_cifar10_cutoff_k%a-%A.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/cutoff_seeds345/cifar10/nn1/local_sigma/maxherding_cifar10_cutoff_k%a-%A.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/maxherding/cutoff_seeds345/cifar10/nn1/local_sigma"
SEEDS=(3 4 5)
K_VALUES=(10 60 480 1440 2000 3500 5000 10000)
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / ${#K_VALUES[@]}))
K_INDEX=$((SLURM_ARRAY_TASK_ID % ${#K_VALUES[@]}))
SEED=${SEEDS[$SEED_INDEX]}
K=${K_VALUES[$K_INDEX]}
EXP_NAME=maxherding_cutoff_cifar10_seed${SEED}_nn1_local_sigma_k${K}

mkdir -p "$OUT_DIR"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
cd deep_al/tools

python train_al.py \
  --cfg ../configs/cifar10/al/RESNET18.yaml \
  --al max_herding \
  --diff_method max_herding \
  --kernel_type rbf \
  --rbf_cutoff_mode probcover \
  --initial_delta 0.75 \
  --initial_sigma 1.0 \
  --local_sigma \
  --local_sigma_k ${K} \
  --sparse_K \
  --matrix_dtype float16 \
  --exp-name ${EXP_NAME} \
  --initial_size 0 \
  --budget 10 \
  --max_iter 9 \
  --seed ${SEED} \
  --eval_model_type NN1
