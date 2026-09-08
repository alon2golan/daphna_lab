#!/bin/bash
#SBATCH --job-name=mh_c100_dn_decay
#SBATCH --mem=96g
#SBATCH -c 4
#SBATCH --time=1-06:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --array=0-11%2
#SBATCH --output=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/density_norm/cifar100/nn1_decay_tuned/mh_cifar100_density_norm_decay_tuned_%a-%A.txt
#SBATCH --error=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/density_norm/cifar100/nn1_decay_tuned/mh_cifar100_density_norm_decay_tuned_%a-%A.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/maxherding/density_norm/cifar100/nn1_decay_tuned"

SEEDS=(3 4)
DEGREE_TYPE=weighted

# Format: alpha|schedule|steps|tag
# CIFAR100 showed alpha=0.5 is useful mainly up to about 400 labels and then
# becomes neutral or harmful. These schedules test abrupt shutoff, gradual
# shutoff, and a middle phase with weaker normalization.
SCHEDULE_CONFIGS=(
  "0.5|early_only|4|a05_early4"
  "0.5|linear_decay_steps|5|a05_decay5"
  "0.5|three_phase|4|a05_three4"
)
SIGMA_CONFIGS=(global k1440)

NUM_SIGMA=${#SIGMA_CONFIGS[@]}
NUM_SCHEDULE=${#SCHEDULE_CONFIGS[@]}
TASKS_PER_SEED=$((NUM_SIGMA * NUM_SCHEDULE))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / TASKS_PER_SEED))
LOCAL_TASK_ID=$((SLURM_ARRAY_TASK_ID % TASKS_PER_SEED))
SIGMA_INDEX=$((LOCAL_TASK_ID % NUM_SIGMA))
SCHEDULE_INDEX=$((LOCAL_TASK_ID / NUM_SIGMA))

SEED=${SEEDS[$SEED_INDEX]}
SIGMA_CONFIG=${SIGMA_CONFIGS[$SIGMA_INDEX]}
IFS='|' read -r BASE_ALPHA SCHEDULE SCHEDULE_STEPS SCHEDULE_TAG <<< "${SCHEDULE_CONFIGS[$SCHEDULE_INDEX]}"

mkdir -p "$OUT_DIR"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
cd deep_al/tools

COMMON_ARGS=(
  --cfg ../configs/cifar100/al/RESNET18.yaml
  --al max_herding
  --diff_method max_herding
  --kernel_type rbf
  --rbf_cutoff_mode probcover
  --initial_delta 0.65
  --initial_sigma 1.0
  --sparse_K
  --matrix_dtype float16
  --initial_size 0
  --budget 100
  --max_iter 9
  --seed "${SEED}"
  --eval_model_type NN1
  --density_normalize_kernel
  --density_norm_alpha "${BASE_ALPHA}"
  --density_norm_mode symmetric
  --density_norm_degree "${DEGREE_TYPE}"
  --density_norm_schedule "${SCHEDULE}"
  --density_norm_schedule_steps "${SCHEDULE_STEPS}"
)

if [[ "$SIGMA_CONFIG" == "global" ]]; then
  EXP_NAME=maxherding_dnorm_decay_cifar100_seed${SEED}_nn1_global_${SCHEDULE_TAG}_${DEGREE_TYPE}
  python train_al.py "${COMMON_ARGS[@]}" --exp-name "${EXP_NAME}"
else
  K=${SIGMA_CONFIG#k}
  EXP_NAME=maxherding_dnorm_decay_cifar100_seed${SEED}_nn1_local_sigma_k${K}_${SCHEDULE_TAG}_${DEGREE_TYPE}
  python train_al.py "${COMMON_ARGS[@]}" --exp-name "${EXP_NAME}" --local_sigma --local_sigma_k "${K}"
fi
