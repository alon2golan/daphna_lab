#!/bin/bash
#SBATCH --job-name=mh_c10_k_ultra
#SBATCH --mem=64g
#SBATCH -c 4
#SBATCH --time=24:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --array=0-7
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/low_budget_k_sweep/cifar10/local_sigma/ultra_wide_k_sweep/maxherding_cifar10_local_sigma_ultra_k%a_seed1-%A.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/low_budget_k_sweep/cifar10/local_sigma/ultra_wide_k_sweep/maxherding_cifar10_local_sigma_ultra_k%a_seed1-%A.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/maxherding/low_budget_k_sweep/cifar10/local_sigma/ultra_wide_k_sweep"
SEED=${SEED_OVERRIDE:-1}
K_VALUES=(240 480 720 960 1440 1920 2880 3840)
K=${K_VALUES[$SLURM_ARRAY_TASK_ID]}
EXP_NAME=low_budget_maxherding_cifar10_local_sigma_ultra_k${K}_seed${SEED}

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
  --local_sigma \
  --local_sigma_k ${K} \
  --exp-name ${EXP_NAME} \
  --initial_size 0 \
  --budget 10 \
  --max_iter 9 \
  --seed ${SEED} \
  --eval_model_type NN1
