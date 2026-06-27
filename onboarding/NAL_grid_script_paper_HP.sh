#!/bin/bash
#SBATCH --mem=30g
#SBATCH -c 8
#SBATCH --time=48:00:00
#SBATCH --gres=gg:g0:1,vmem:30g
#SBATCH --mail-user=tom.novak@mail.huji.ac.il
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=gpu_grid_noise_al_%j.out

set -euo pipefail

REPO_DIR="/cs/labs/daphna/$USER/projects/daphna_lab"
cd "$REPO_DIR"
source "$REPO_DIR/venv/bin/activate"
cd "$REPO_DIR/deep_al/tools"

CFG_PATH="../configs/cifar10/al/RESNET18.yaml"
SEED=0
EXP_NAME="grid_NAL_paper_hp"
CIFAR_N_VARIANT="auto"
GRID_OUT_DIR="$REPO_DIR/grid_outputs/paper_hp_NAL_grid"
mkdir -p "$GRID_OUT_DIR"

# Keep CIFAR-100N out of this CIFAR-10 config job. Run a separate CIFAR-100 config for cifar100n.
GRID_NOISE_TYPES="clean,sym,asym,cifar10n"

# Only methods with clear original/default HP are included here.
# MISP variants are excluded because I could not find a public original HP source.
GRID_AL_METHODS="typiclust,probcover,dcom,maxherding,margin,coreset,random"

# Only implemented methods in run_noise_al_pipeline.py.
GRID_NOISE_METHODS="none,label_cleaning"

python3 run_noise_al_pipeline.py \
  --cfg "$CFG_PATH" \
  --grid_noise_types "$GRID_NOISE_TYPES" \
  --grid_al_methods "$GRID_AL_METHODS" \
  --grid_noise_methods "$GRID_NOISE_METHODS" \
  --grid_out_dir "$GRID_OUT_DIR" \
  --noise_rate 0.2 \
  --cifar_n_variant "$CIFAR_N_VARIANT" \
  --seed "$SEED" \
  --exp_name "$EXP_NAME" \
  --paper_hparams

# For a quick smoke test, add:
#   --max_epoch 5 --max_iter 1 --budget 10
