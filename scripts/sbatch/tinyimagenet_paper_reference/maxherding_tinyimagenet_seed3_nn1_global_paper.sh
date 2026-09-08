#!/bin/bash
# Paper-style Tiny ImageNet MaxHerding reference.
#
# This intentionally does NOT use:
# - ProbCover-style RBF cutoff
# - local sigma
# - density normalization
# - sparse_K
#
# The goal is to reproduce the MaxHerding paper reference as closely as the
# current codebase allows: smooth global RBF MaxHerding with 1NN evaluation on
# fixed SSL features, 200 labels per round, 10 reported labeled-set sizes.

#SBATCH --job-name=mh_tiny_s3_paper
#SBATCH --mem=180g
#SBATCH -c 4
#SBATCH --time=3-00:00:00
#SBATCH --gres=gg:g4:1
#SBATCH --output=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/reference_validation/maxherding/tinyimagenet/paper_nn1/maxherding_tinyimagenet_seed3_paper_nn1-%j.txt
#SBATCH --error=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/reference_validation/maxherding/tinyimagenet/paper_nn1/maxherding_tinyimagenet_seed3_paper_nn1-%j.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/reference_validation/maxherding/tinyimagenet/paper_nn1"
SEED=${SEED_OVERRIDE:-3}
EXP_NAME=refval_maxherding_tinyimagenet_seed${SEED}_paper_nn1_global

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
  --exp-name ${EXP_NAME} \
  --initial_size 0 \
  --budget 200 \
  --max_iter 9 \
  --seed ${SEED} \
  --eval_model_type NN1
