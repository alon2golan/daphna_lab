#!/bin/bash
#SBATCH --job-name=mh_c10_delta_prof
#SBATCH --mem=64g
#SBATCH -c 4
#SBATCH --time=24:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --array=0-59%4
#SBATCH --output=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/professor_notes_2026_05_28/cifar10_delta_sensitivity/mh_cifar10_delta_%a-%A.txt
#SBATCH --error=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/professor_notes_2026_05_28/cifar10_delta_sensitivity/mh_cifar10_delta_%a-%A.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/professor_notes_2026_05_28/cifar10_delta_sensitivity"

SEEDS=(3 4 5)
DELTAS=(0.65 0.70 0.75 0.80 0.85)
ALPHAS=(0 0.5)
SIGMA_CONFIGS=(global k3500)

NUM_SIGMA=${#SIGMA_CONFIGS[@]}
NUM_ALPHAS=${#ALPHAS[@]}
NUM_DELTAS=${#DELTAS[@]}
TASKS_PER_SEED=$((NUM_SIGMA * NUM_ALPHAS * NUM_DELTAS))

SEED_INDEX=$((SLURM_ARRAY_TASK_ID / TASKS_PER_SEED))
LOCAL_TASK=$((SLURM_ARRAY_TASK_ID % TASKS_PER_SEED))
DELTA_INDEX=$((LOCAL_TASK / (NUM_SIGMA * NUM_ALPHAS)))
REST=$((LOCAL_TASK % (NUM_SIGMA * NUM_ALPHAS)))
ALPHA_INDEX=$((REST / NUM_SIGMA))
SIGMA_INDEX=$((REST % NUM_SIGMA))

SEED=${SEEDS[$SEED_INDEX]}
DELTA=${DELTAS[$DELTA_INDEX]}
ALPHA=${ALPHAS[$ALPHA_INDEX]}
SIGMA_CONFIG=${SIGMA_CONFIGS[$SIGMA_INDEX]}
DELTA_TAG=${DELTA/./p}

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
  --initial_delta "${DELTA}"
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
  --density_norm_degree weighted
)

if [[ "$SIGMA_CONFIG" == "global" ]]; then
  EXP_NAME=profnotes_delta_cifar10_seed${SEED}_nn1_global_delta${DELTA_TAG}_alpha${ALPHA}_weighted
  python train_al.py "${COMMON_ARGS[@]}" --exp-name "${EXP_NAME}"
else
  K=${SIGMA_CONFIG#k}
  EXP_NAME=profnotes_delta_cifar10_seed${SEED}_nn1_local_k${K}_delta${DELTA_TAG}_alpha${ALPHA}_weighted
  python train_al.py "${COMMON_ARGS[@]}" --exp-name "${EXP_NAME}" --local_sigma --local_sigma_k "${K}"
fi
