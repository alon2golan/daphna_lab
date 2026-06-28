#!/bin/bash
#SBATCH --mem=30g
#SBATCH -c 8
#SBATCH --time=48:00:00
#SBATCH --gres=gg:g0:1,vmem:30g
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

GRID_NOISE_TYPES="clean,sym,asym,cifar100n"
GRID_AL_METHODS="typiclust,probcover,dcom,maxherding,margin,coreset,random"
GRID_NOISE_METHODS="none,label_cleaning"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "PYTHON=$(which python3)"
echo "CFG_PATH=$CFG_PATH"
echo "GRID_NOISE_TYPES=$GRID_NOISE_TYPES"
echo "GRID_AL_METHODS=$GRID_AL_METHODS"
echo "GRID_NOISE_METHODS=$GRID_NOISE_METHODS"
echo "GRID_OUT_DIR=$GRID_OUT_DIR"
echo "Started at $(date)"

python3 -u run_noise_al_pipeline.py \
  --cfg "$CFG_PATH" \
  --grid_noise_types "$GRID_NOISE_TYPES" \
  --grid_al_methods "$GRID_AL_METHODS" \
  --grid_noise_methods "$GRID_NOISE_METHODS" \
  --grid_out_dir "$GRID_OUT_DIR" \
  --noise_rate 0.2 \
  --cifar_n_variant "$CIFAR_N_VARIANT" \
  --seed "$SEED" \
  --exp_name "$EXP_NAME" \
  --eval_model_type from_features \
    --also_eval_knn \
  --paper_hparams

echo "Finished at $(date)"
