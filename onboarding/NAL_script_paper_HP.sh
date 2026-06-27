#!/bin/bash
#SBATCH --mem=20g
#SBATCH -c 6
#SBATCH --time=24:00:00
#SBATCH --gres=gg:g0:1,vmem:20g
#SBATCH --mail-user=tom.novak@mail.huji.ac.il
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=gpu_noise_al_%j.log
#SBATCH --error=gpu_noise_al_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_DIR="/cs/labs/daphna/$USER/projects/daphna_lab"
cd "$REPO_DIR"
source "$REPO_DIR/venv/bin/activate"
cd "$REPO_DIR/deep_al/tools"

CFG_PATH="../configs/cifar100/al/RESNET18.yaml"
SEED=0
EXP_NAME="NAL_paper_hp_typiclust_cifar100"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "PYTHON=$(which python3)"
echo "CFG_PATH=$CFG_PATH"
echo "SEED=$SEED"
echo "EXP_NAME=$EXP_NAME"
echo "Started at $(date)"

python3 -u run_noise_al_pipeline.py \
  --cfg "$CFG_PATH" \
  --noise_type sym \
  --noise_rate 0.2 \
  --al_method typiclust \
  --noise_method label_cleaning \
  --seed "$SEED" \
  --exp_name "$EXP_NAME" \
  --eval_model_type from_features \
  --paper_hparams

echo "Finished at $(date)"
