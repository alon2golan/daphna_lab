#!/bin/bash
#SBATCH --job-name=mh_c100_dn_sched
#SBATCH --mem=96g
#SBATCH -c 4
#SBATCH --time=1-06:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --array=0-5%2
#SBATCH --output=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/density_norm/cifar100/nn1_schedule/mh_cifar100_density_norm_schedule_%a-%A.txt
#SBATCH --error=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/density_norm/cifar100/nn1_schedule/mh_cifar100_density_norm_schedule_%a-%A.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/maxherding/density_norm/cifar100/nn1_schedule"

SEED=3
BASE_ALPHA=0.5
DEGREE_TYPE=weighted
SCHEDULES=(linear_decay early_only three_phase)
SIGMA_CONFIGS=(global k1440)

NUM_SIGMA=${#SIGMA_CONFIGS[@]}
SIGMA_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SIGMA))
SCHEDULE_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SIGMA))

SIGMA_CONFIG=${SIGMA_CONFIGS[$SIGMA_INDEX]}
SCHEDULE=${SCHEDULES[$SCHEDULE_INDEX]}

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
  --density_norm_schedule_steps 3
)

if [[ "$SIGMA_CONFIG" == "global" ]]; then
  EXP_NAME=maxherding_dnorm_sched_cifar100_seed${SEED}_nn1_global_alpha${BASE_ALPHA}_${DEGREE_TYPE}_${SCHEDULE}
  python train_al.py "${COMMON_ARGS[@]}" --exp-name "${EXP_NAME}"
else
  K=${SIGMA_CONFIG#k}
  EXP_NAME=maxherding_dnorm_sched_cifar100_seed${SEED}_nn1_local_sigma_k${K}_alpha${BASE_ALPHA}_${DEGREE_TYPE}_${SCHEDULE}
  python train_al.py "${COMMON_ARGS[@]}" --exp-name "${EXP_NAME}" --local_sigma --local_sigma_k "${K}"
fi
