#!/bin/bash
#SBATCH --mem=40g
#SBATCH -c 8
#SBATCH --time=72:00:00
#SBATCH --gres=gg:g0:1,vmem:40g
#SBATCH --mail-user=tom.novak@mail.huji.ac.il
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=gpu_grid_noise_al_%j.log
#SBATCH --error=gpu_grid_noise_al_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_DIR="/cs/labs/daphna/$USER/projects/daphna_lab"
cd "$REPO_DIR"
source "$REPO_DIR/venv/bin/activate"
cd "$REPO_DIR/deep_al/tools"

CFG_PATH="../configs/cifar100/al/RESNET18.yaml"
SEED=0
EXP_NAME="grid_NAL_paper_hp"
CIFAR_N_VARIANT="auto"
GRID_OUT_DIR="$REPO_DIR/grid_outputs/paper_hp_NAL_grid"
mkdir -p "$GRID_OUT_DIR"

# CIFAR100 only
GRID_NOISE_TYPES="clean,sym,asym,cifar100n"
GRID_AL_METHODS="typiclust,probcover,dcom,maxherding,margin,coreset,random"
GRID_NOISE_METHODS="none,label_cleaning"

NOISE_RATE=0.2

# Fair final label budget for all methods:
# initial 100 + 4 AL rounds * 100 = 500 final labeled samples
INITIAL_SIZE=100
BUDGET_SIZE=100
MAX_ITER=4

# Optional: leave empty to let run_noise_al_pipeline.py build CIFAR100 groups from torchvision
SUPERCLASS_GROUPS_PATH=""

EXTRA_NOISE_ARGS=()
if [[ -n "$SUPERCLASS_GROUPS_PATH" ]]; then
  EXTRA_NOISE_ARGS+=(--superclass_groups_path "$SUPERCLASS_GROUPS_PATH")
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "PYTHON=$(which python3)"
echo "CFG_PATH=$CFG_PATH"
echo "GRID_NOISE_TYPES=$GRID_NOISE_TYPES"
echo "GRID_AL_METHODS=$GRID_AL_METHODS"
echo "GRID_NOISE_METHODS=$GRID_NOISE_METHODS"
echo "GRID_OUT_DIR=$GRID_OUT_DIR"
echo "NOISE_RATE=$NOISE_RATE"
echo "INITIAL_SIZE=$INITIAL_SIZE"
echo "BUDGET_SIZE=$BUDGET_SIZE"
echo "MAX_ITER=$MAX_ITER"
echo "SUPERCLASS_GROUPS_PATH=$SUPERCLASS_GROUPS_PATH"
echo "Started at $(date)"

python3 -u run_noise_al_pipeline.py \
  --cfg "$CFG_PATH" \
  --grid_noise_types "$GRID_NOISE_TYPES" \
  --grid_al_methods "$GRID_AL_METHODS" \
  --grid_noise_methods "$GRID_NOISE_METHODS" \
  --grid_out_dir "$GRID_OUT_DIR" \
  --noise_rate "$NOISE_RATE" \
  "${EXTRA_NOISE_ARGS[@]}" \
  --cifar_n_variant "$CIFAR_N_VARIANT" \
  --seed "$SEED" \
  --exp_name "$EXP_NAME" \
  --eval_model_type from_features \
  --also_eval_knn \
  --knn_k 1 \
  --paper_hparams \
  --initial_size "$INITIAL_SIZE" \
  --budget "$BUDGET_SIZE" \
  --max_iter "$MAX_ITER"

echo "Finished at $(date)"