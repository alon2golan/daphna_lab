#!/bin/bash
#SBATCH --mem=20g
#SBATCH -c 6
#SBATCH --time=24:00:00
#SBATCH --gres=gg:g0:1,vmem:20g
#SBATCH --mail-user=tom.novak@mail.huji.ac.il
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=gpu_noise_al_%j.out

set -euo pipefail

REPO_DIR="/cs/labs/daphna/$USER/projects/daphna_lab"
cd "$REPO_DIR"
source "$REPO_DIR/venv/bin/activate"
cd "$REPO_DIR/deep_al/tools"

CFG_PATH="../configs/cifar10/al/RESNET18.yaml"
SEED=0
EXP_NAME="NAL_paper_hp_typiclust_cifar10"

python3 run_noise_al_pipeline.py \
  --cfg "$CFG_PATH" \
  --noise_type sym \
  --noise_rate 0.2 \
  --al_method typiclust \
  --noise_method label_cleaning \
  --seed "$SEED" \
  --exp_name "$EXP_NAME" \
  --paper_hparams

# For a quick debug run, add for example:
#   --max_epoch 5 --max_iter 1 --budget 10
