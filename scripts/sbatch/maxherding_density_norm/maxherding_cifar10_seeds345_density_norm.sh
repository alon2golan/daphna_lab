#!/bin/bash
#SBATCH --job-name=mh_c10_dnorm
#SBATCH --mem=64g
#SBATCH -c 4
#SBATCH --time=24:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --array=0-11%2
#SBATCH --output=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/density_norm/cifar10/nn1/mh_cifar10_density_norm_%a-%A.txt
#SBATCH --error=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/density_norm/cifar10/nn1/mh_cifar10_density_norm_%a-%A.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/maxherding/density_norm/cifar10/nn1"

SEEDS=(3)
ALPHAS=(0 0.25 0.5)
DEGREE_TYPES=(effective weighted)
SIGMA_CONFIGS=(global k3500)

NUM_SIGMA=${#SIGMA_CONFIGS[@]}
NUM_DEGREES=${#DEGREE_TYPES[@]}
NUM_ALPHAS=${#ALPHAS[@]}

SIGMA_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SIGMA))
DEGREE_INDEX=$(((SLURM_ARRAY_TASK_ID / NUM_SIGMA) % NUM_DEGREES))
ALPHA_INDEX=$(((SLURM_ARRAY_TASK_ID / (NUM_SIGMA * NUM_DEGREES)) % NUM_ALPHAS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / (NUM_SIGMA * NUM_DEGREES * NUM_ALPHAS)))

SIGMA_CONFIG=${SIGMA_CONFIGS[$SIGMA_INDEX]}
DEGREE_TYPE=${DEGREE_TYPES[$DEGREE_INDEX]}
ALPHA=${ALPHAS[$ALPHA_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

mkdir -p "$OUT_DIR"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
cd deep_al/tools

COMMON_ARGS=(
  --cfg ../configs/cifar10/al/RESNET18.yaml
  --al max_herding
  --diff_method max_herding
  --kernel_type rbf
  --rbf_cutoff_mode probcover
  --initial_delta 0.75
  --initial_sigma 1.0
  --sparse_K
  --matrix_dtype float16
  --initial_size 0
  --budget 10
  --max_iter 9
  --seed "${SEED}"
  --eval_model_type NN1
  --density_normalize_kernel
  --density_norm_alpha "${ALPHA}"
  --density_norm_mode symmetric
  --density_norm_degree "${DEGREE_TYPE}"
)

if [[ "$SIGMA_CONFIG" == "global" ]]; then
  EXP_NAME=maxherding_dnorm_cifar10_seed${SEED}_nn1_global_alpha${ALPHA}_${DEGREE_TYPE}
  python train_al.py "${COMMON_ARGS[@]}" --exp-name "${EXP_NAME}"
else
  K=${SIGMA_CONFIG#k}
  EXP_NAME=maxherding_dnorm_cifar10_seed${SEED}_nn1_local_sigma_k${K}_alpha${ALPHA}_${DEGREE_TYPE}
  python train_al.py "${COMMON_ARGS[@]}" --exp-name "${EXP_NAME}" --local_sigma --local_sigma_k "${K}"
fi
