#!/bin/bash
# Tiny ImageNet MaxHerding sanity run.
#
# This is the first diagnostic step after the low Tiny ImageNet accuracy:
# keep the runnable sparse RBF path, but remove the ProbCover-style cutoff.
#
# Compared to the cutoff run, this removes:
# - --rbf_cutoff_mode probcover
#
# It keeps:
# - --sparse_K
# - global sigma = 1.0
# - 1NN evaluation on fixed SSL features
# - budget 200 per round

#SBATCH --job-name=mh_tiny_s3_nocut
#SBATCH --mem=128g
#SBATCH -c 4
#SBATCH --time=3-00:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --output=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/reference_validation/maxherding/tinyimagenet/no_cutoff_sparse/maxherding_tinyimagenet_seed3_no_cutoff_sparse-%j.txt
#SBATCH --error=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/reference_validation/maxherding/tinyimagenet/no_cutoff_sparse/maxherding_tinyimagenet_seed3_no_cutoff_sparse-%j.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/reference_validation/maxherding/tinyimagenet/no_cutoff_sparse"
SEED=${SEED_OVERRIDE:-3}
EXP_NAME=refval_maxherding_tinyimagenet_seed${SEED}_nn1_no_probcover_cutoff_sparse

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
  --sparse_K \
  --matrix_dtype float16 \
  --exp-name ${EXP_NAME} \
  --initial_size 0 \
  --budget 200 \
  --max_iter 9 \
  --seed ${SEED} \
  --eval_model_type NN1
