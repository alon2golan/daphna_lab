#!/bin/bash
# Paper-style Tiny ImageNet MaxHerding reference using the stronger Maor feature pair.
#
# Sanity check before running:
# maor_train_features.npy / maor_test_features.npy gives about 29.94% full-train
# 1NN accuracy with val_annotations labels, while the previous default feature
# pair gave only about 13.13%.

#SBATCH --job-name=mh_tiny_s3_maor
#SBATCH --mem=180g
#SBATCH -c 4
#SBATCH --time=3-00:00:00
#SBATCH --gres=gg:g4:1
#SBATCH --output=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/reference_validation/maxherding/tinyimagenet/paper_nn1_maor/maxherding_tinyimagenet_seed3_paper_nn1_maor-%j.txt
#SBATCH --error=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/reference_validation/maxherding/tinyimagenet/paper_nn1_maor/maxherding_tinyimagenet_seed3_paper_nn1_maor-%j.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/reference_validation/maxherding/tinyimagenet/paper_nn1_maor"
SEED=${SEED_OVERRIDE:-3}
EXP_NAME=refval_maxherding_tinyimagenet_seed${SEED}_paper_nn1_global_maor_features

mkdir -p "$OUT_DIR"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
cd deep_al/tools

python train_al.py \
  --cfg ../configs/tinyimagenet/al/RESNET18_LOW_BUDGET.yaml \
  --al max_herding \
  --diff_method max_herding \
  --kernel_type rbf \
  --initial_delta 1.0 \
  --initial_sigma 1.0 \
  --matrix_dtype float16 \
  --features_tag maor \
  --exp-name ${EXP_NAME} \
  --initial_size 0 \
  --budget 200 \
  --max_iter 9 \
  --seed ${SEED} \
  --eval_model_type NN1
