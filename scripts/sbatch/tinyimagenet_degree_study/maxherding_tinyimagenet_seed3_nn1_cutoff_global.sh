#!/bin/bash
#SBATCH --job-name=mh_tiny_s3_glob
#SBATCH --mem=128g
#SBATCH -c 4
#SBATCH --time=3-00:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --output=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/tinyimagenet_degree_study/global/maxherding_tinyimagenet_seed3_global-%j.txt
#SBATCH --error=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/tinyimagenet_degree_study/global/maxherding_tinyimagenet_seed3_global-%j.err
#SBATCH --killable

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/maxherding/tinyimagenet_degree_study/global"
DELTA_FILE="$PROJECT_DIR/results_summaries/maxherding_tinyimagenet_density_selection/tinyimagenet_delta_seed3.txt"
SEED=3
EXP_NAME=maxherding_tinyimagenet_degree_seed${SEED}_nn1_cutoff_global_sigma

set -euo pipefail
mkdir -p "$OUT_DIR"

if [[ -z "${DELTA_OVERRIDE:-}" ]]; then
  if [[ ! -f "$DELTA_FILE" ]]; then
    echo "Missing delta file: $DELTA_FILE"
    echo "Run tinyimagenet_delta_probe_seed3.sh first, or pass DELTA_OVERRIDE=<value>."
    exit 1
  fi
  DELTA_OVERRIDE=$(awk -F= '/^delta=/{print $2; exit}' "$DELTA_FILE")
fi
echo "Using DELTA_OVERRIDE=${DELTA_OVERRIDE}"

cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
cd deep_al/tools

python train_al.py \
  --cfg ../configs/tinyimagenet/al/RESNET18_LOW_BUDGET.yaml \
  --al max_herding \
  --diff_method max_herding \
  --kernel_type rbf \
  --rbf_cutoff_mode probcover \
  --initial_delta ${DELTA_OVERRIDE} \
  --initial_sigma 1.0 \
  --sparse_K \
  --matrix_dtype float16 \
  --exp-name ${EXP_NAME} \
  --initial_size 0 \
  --budget 200 \
  --max_iter 9 \
  --seed ${SEED} \
  --eval_model_type NN1
