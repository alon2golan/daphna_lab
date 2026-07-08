#!/bin/bash
#SBATCH --mem=30g
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH --gres=gg:g0:1,vmem:30g
#SBATCH --mail-user=tom.novak@mail.huji.ac.il
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=gpu_grid_noise_al_smoke_%j.log
#SBATCH --error=gpu_grid_noise_al_smoke_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_DIR="/cs/labs/daphna/$USER/projects/daphna_lab"
cd "$REPO_DIR"
source "$REPO_DIR/venv/bin/activate"
cd "$REPO_DIR/deep_al/tools"

CFG_PATH="../configs/cifar100/al/RESNET18.yaml"
SEED=0
EXP_NAME="smoke_new_noise_methods"
BASE_GRID_OUT_DIR="$REPO_DIR/grid_outputs/smoke_new_noise_methods"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
DEBUG_JSON="${DEBUG_JSON:-0}"
GRID_JSON_PATH="${GRID_JSON_PATH:-$BASE_GRID_OUT_DIR/aggregate_results_${RUN_TIMESTAMP}.json}"
mkdir -p "$BASE_GRID_OUT_DIR"

GRID_NOISE_TYPES="sym"
GRID_AL_METHODS="typiclust"
NOISE_RATE=0.2
INITIAL_SIZE=100
BUDGET_SIZE=100
MAX_ITER=1

JSON_ARGS=(--grid_json_path "$GRID_JSON_PATH")
if [[ "$DEBUG_JSON" == "1" || "$DEBUG_JSON" == "true" || "$DEBUG_JSON" == "TRUE" ]]; then
  JSON_ARGS+=(--debug_json)
fi

COMMON_ARGS=(
  --cfg "$CFG_PATH"
  --grid_noise_types "$GRID_NOISE_TYPES"
  --grid_al_methods "$GRID_AL_METHODS"
  --noise_rate "$NOISE_RATE"
  --seed "$SEED"
  --eval_model_type from_features
  --also_eval_knn
  --knn_k 1
  --paper_hparams
  --initial_size "$INITIAL_SIZE"
  --budget "$BUDGET_SIZE"
  --max_iter "$MAX_ITER"
  --warmup_episode 0
  --warmup_min_acc 0
  "${JSON_ARGS[@]}"
)

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "PYTHON=$(which python3)"
echo "CFG_PATH=$CFG_PATH"
echo "GRID_NOISE_TYPES=$GRID_NOISE_TYPES"
echo "GRID_AL_METHODS=$GRID_AL_METHODS"
echo "BASE_GRID_OUT_DIR=$BASE_GRID_OUT_DIR"
echo "GRID_JSON_PATH=$GRID_JSON_PATH"
echo "DEBUG_JSON=$DEBUG_JSON"
echo "Started new-methods smoke test at $(date)"

python3 -u run_noise_al_pipeline.py \
  "${COMMON_ARGS[@]}" \
  --exp_name "${EXP_NAME}_cleaning" \
  --grid_out_dir "$BASE_GRID_OUT_DIR/cleaning" \
  --grid_noise_methods "crossval,temporal,aum" \
  --loss_fun cross_entropy \
  --cleaning_conf_threshold 0.8 \
  --cleaning_pred_source checkpoint \
  --crossval_folds 2 \
  --crossval_max_iter 100 \
  --temporal_momentum 0.8 \
  --temporal_min_seen 1 \
  --temporal_conf_threshold 0.8 \
  --aum_percentile 10 \
  --aum_conf_threshold 0.8

for LOSS_FUN in gce bootstrap_soft bootstrap_hard; do
  python3 -u run_noise_al_pipeline.py \
    "${COMMON_ARGS[@]}" \
    --exp_name "${EXP_NAME}_${LOSS_FUN}" \
    --grid_out_dir "$BASE_GRID_OUT_DIR/robust_${LOSS_FUN}" \
    --grid_noise_methods "none" \
    --loss_fun "$LOSS_FUN" \
    --gce_q 0.7 \
    --bootstrap_beta 0.8
done

echo "Finished new-methods smoke test at $(date)"
echo "Aggregate JSON: $GRID_JSON_PATH"
