#!/bin/bash
#SBATCH --mem=20g
#SBATCH -c 6
#SBATCH --time=6:00:00
#SBATCH --gres=gg:g0:1,vmem:20g
#SBATCH --mail-user=tom.novak@mail.huji.ac.il
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=gpu_grid_noise_al_%A_%a.out
#SBATCH --array=0-29

set -euo pipefail

repo_dir="/cs/labs/daphna/${USER}/projects/daphna_lab"
cd "$repo_dir"
source "$repo_dir/venv/bin/activate"
cd "$repo_dir/deep_al/tools"

CFG_PATH="../configs/cifar10/al/RESNET18.yaml"
SEED=0
BUDGET=100
INITIAL_SIZE=100
MAX_ITER=4
EXP_NAME_PREFIX="grid_NAL_cifar10"
INITIAL_DELTA=0.6
EVAL_MODEL_TYPE="from_images"
MAX_EPOCH=50
WARMUP_EPISODE=3
WARMUP_MIN_ACC=30.0
CIFAR_N_VARIANT="auto"

# CIFAR10 config: do not include cifar100n here.
NOISE_TYPES=("sym" "asym" "cifar10n")
AL_METHODS=("typiclust" "probcover" "margin" "dcom" "maxherding")
NOISE_METHODS=("none" "label_cleaning")

num_noise=${#NOISE_TYPES[@]}
num_al=${#AL_METHODS[@]}
num_nm=${#NOISE_METHODS[@]}
total=$((num_noise * num_al * num_nm))
idx=${SLURM_ARRAY_TASK_ID:-0}

if (( idx < 0 || idx >= total )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=$idx, expected 0-$((total - 1))"
  exit 1
fi

nm_idx=$((idx % num_nm))
al_idx=$(((idx / num_nm) % num_al))
noise_idx=$((idx / (num_nm * num_al)))

NOISE_TYPE=${NOISE_TYPES[$noise_idx]}
AL_METHOD=${AL_METHODS[$al_idx]}
NOISE_METHOD=${NOISE_METHODS[$nm_idx]}

KERNEL_TYPE="rbf"
EXTRA_ARGS=()
if [[ "$AL_METHOD" == "probcover" ]]; then
  KERNEL_TYPE="tophat"
  EXTRA_ARGS+=("--sparse_K")
fi

EXP_NAME="${EXP_NAME_PREFIX}_${NOISE_TYPE}_${AL_METHOD}_${NOISE_METHOD}_seed${SEED}"
GRID_OUT_DIR="$repo_dir/grid_outputs/full_NAL_grid_cifar10"
mkdir -p "$GRID_OUT_DIR"

echo "Running idx=$idx/$((total - 1)): noise=$NOISE_TYPE al=$AL_METHOD noise_method=$NOISE_METHOD"

python3 run_noise_al_pipeline.py \
  --cfg "$CFG_PATH" \
  --noise_type "$NOISE_TYPE" \
  --noise_rate 0.2 \
  --al_method "$AL_METHOD" \
  --noise_method "$NOISE_METHOD" \
  --grid_out_dir "$GRID_OUT_DIR" \
  --cifar_n_variant "$CIFAR_N_VARIANT" \
  --seed "$SEED" \
  --budget "$BUDGET" \
  --initial_size "$INITIAL_SIZE" \
  --max_iter "$MAX_ITER" \
  --exp_name "$EXP_NAME" \
  --initial_delta "$INITIAL_DELTA" \
  --kernel_type "$KERNEL_TYPE" \
  --eval_model_type "$EVAL_MODEL_TYPE" \
  --max_epoch "$MAX_EPOCH" \
  --warmup_episode "$WARMUP_EPISODE" \
  --warmup_min_acc "$WARMUP_MIN_ACC" \
  "${EXTRA_ARGS[@]}"
