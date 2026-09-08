#!/bin/bash
# Paper-style Tiny ImageNet MaxHerding reference using the stronger Maor feature pair.
#
# This version uses VAL_RATIO=0.0 so MaxHerding selects from the full Tiny
# ImageNet training pool instead of reserving 5% as validation. For 1NN feature
# evaluation, the validation split is not needed for model selection.

#SBATCH --job-name=mh_tiny_s3_maor_full
#SBATCH --mem=180g
#SBATCH -c 4
#SBATCH --time=3-00:00:00
#SBATCH --gres=gg:g4:1
#SBATCH --output=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/reference_validation/maxherding/tinyimagenet/paper_nn1_maor/maxherding_tinyimagenet_seed3_paper_nn1_maor_no_val-%j.txt
#SBATCH --error=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/reference_validation/maxherding/tinyimagenet/paper_nn1_maor/maxherding_tinyimagenet_seed3_paper_nn1_maor_no_val-%j.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/reference_validation/maxherding/tinyimagenet/paper_nn1_maor"
SEED=${SEED_OVERRIDE:-3}
EXP_NAME=refval_maxherding_tinyimagenet_seed${SEED}_paper_nn1_global_maor_features_no_val

mkdir -p "$OUT_DIR"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
cd deep_al/tools

python train_al.py \
  --cfg ../configs/tinyimagenet/al/RESNET18_LOW_BUDGET_NO_VAL.yaml \
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
