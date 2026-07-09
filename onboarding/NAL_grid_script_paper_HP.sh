#!/bin/bash
#SBATCH --mem=40g
#SBATCH -c 8
#SBATCH --time=96:00:00
#SBATCH --gres=gg:g0:1,vmem:40g
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
EXP_NAME="grid_NAL_paper_hp_aggressive_cleaning"
CIFAR_N_VARIANT="auto"
BASE_GRID_OUT_DIR="$REPO_DIR/grid_outputs/paper_hp_NAL_grid_noise_methods_aggressive_cleaning"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
DEBUG_JSON="${DEBUG_JSON:-0}"
GRID_JSON_PATH="${GRID_JSON_PATH:-$BASE_GRID_OUT_DIR/aggregate_results_${RUN_TIMESTAMP}.json}"
mkdir -p "$BASE_GRID_OUT_DIR"

GRID_NOISE_TYPES="clean,sym,asym,cifar100n"
GRID_AL_METHODS="typiclust,probcover,maxherding,margin,coreset,random"

NOISE_RATE=0.2
INITIAL_SIZE=100
BUDGET_SIZE=100
MAX_ITER=4

# More aggressive cleaning thresholds for the next debug run.
# Override from command line if needed, e.g. AUM_CONF_THRESHOLD=0.6 sbatch ...
LABEL_CLEANING_CONF_THRESHOLD="${LABEL_CLEANING_CONF_THRESHOLD:-0.7}"
TEMPORAL_CONF_THRESHOLD="${TEMPORAL_CONF_THRESHOLD:-0.7}"
TEMPORAL_MIN_SEEN="${TEMPORAL_MIN_SEEN:-1}"
AUM_PERCENTILE="${AUM_PERCENTILE:-20}"
AUM_CONF_THRESHOLD="${AUM_CONF_THRESHOLD:-0.7}"

CIFAR_N_LABELS_PATH="$REPO_DIR/deep_al/tools/data/CIFAR-100_human.pt"
SUPERCLASS_GROUPS_PATH=""

EXTRA_NOISE_ARGS=()
if [[ -n "$SUPERCLASS_GROUPS_PATH" ]]; then
  EXTRA_NOISE_ARGS+=(--superclass_groups_path "$SUPERCLASS_GROUPS_PATH")
fi

EXTRA_CIFAR_N_ARGS=()
if [[ -f "$CIFAR_N_LABELS_PATH" ]]; then
  EXTRA_CIFAR_N_ARGS+=(--cifar_n_labels_path "$CIFAR_N_LABELS_PATH")
else
  echo "WARNING: CIFAR_N_LABELS_PATH does not exist: $CIFAR_N_LABELS_PATH"
  echo "CIFAR100N runs may fail, but the grid should continue."
fi

JSON_ARGS=(--grid_json_path "$GRID_JSON_PATH")
if [[ "$DEBUG_JSON" == "1" || "$DEBUG_JSON" == "true" || "$DEBUG_JSON" == "TRUE" ]]; then
  JSON_ARGS+=(--debug_json)
fi

COMMON_ARGS=(
  --cfg "$CFG_PATH"
  --grid_noise_types "$GRID_NOISE_TYPES"
  --grid_al_methods "$GRID_AL_METHODS"
  --noise_rate "$NOISE_RATE"
  "${EXTRA_NOISE_ARGS[@]}"
  "${EXTRA_CIFAR_N_ARGS[@]}"
  --cifar_n_variant "$CIFAR_N_VARIANT"
  --seed "$SEED"
  --eval_model_type from_features
  --also_eval_knn
  --knn_k 1
  --paper_hparams
  --initial_size "$INITIAL_SIZE"
  --budget "$BUDGET_SIZE"
  --max_iter "$MAX_ITER"
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
echo "LABEL_CLEANING_CONF_THRESHOLD=$LABEL_CLEANING_CONF_THRESHOLD"
echo "TEMPORAL_CONF_THRESHOLD=$TEMPORAL_CONF_THRESHOLD"
echo "TEMPORAL_MIN_SEEN=$TEMPORAL_MIN_SEEN"
echo "AUM_PERCENTILE=$AUM_PERCENTILE"
echo "AUM_CONF_THRESHOLD=$AUM_CONF_THRESHOLD"
echo "NOISE_RATE=$NOISE_RATE"
echo "INITIAL_SIZE=$INITIAL_SIZE"
echo "BUDGET_SIZE=$BUDGET_SIZE"
echo "MAX_ITER=$MAX_ITER"
echo "SUPERCLASS_GROUPS_PATH=$SUPERCLASS_GROUPS_PATH"
echo "CIFAR_N_LABELS_PATH=$CIFAR_N_LABELS_PATH"
echo "Started at $(date)"

CE_GRID_OUT_DIR="$BASE_GRID_OUT_DIR/cleaning_cross_entropy"
mkdir -p "$CE_GRID_OUT_DIR"

python3 -u run_noise_al_pipeline.py \
  "${COMMON_ARGS[@]}" \
  --exp_name "${EXP_NAME}_cleaning_ce" \
  --grid_out_dir "$CE_GRID_OUT_DIR" \
  --grid_noise_methods "none,label_cleaning,crossval,temporal,aum" \
  --loss_fun cross_entropy \
  --cleaning_conf_threshold "$LABEL_CLEANING_CONF_THRESHOLD" \
  --cleaning_pred_source checkpoint \
  --crossval_folds 5 \
  --crossval_max_iter 300 \
  --temporal_momentum 0.8 \
  --temporal_min_seen "$TEMPORAL_MIN_SEEN" \
  --temporal_conf_threshold "$TEMPORAL_CONF_THRESHOLD" \
  --aum_percentile "$AUM_PERCENTILE" \
  --aum_conf_threshold "$AUM_CONF_THRESHOLD"

for LOSS_FUN in gce bootstrap_soft bootstrap_hard; do
  ROBUST_GRID_OUT_DIR="$BASE_GRID_OUT_DIR/robust_${LOSS_FUN}"
  mkdir -p "$ROBUST_GRID_OUT_DIR"

  python3 -u run_noise_al_pipeline.py \
    "${COMMON_ARGS[@]}" \
    --exp_name "${EXP_NAME}_robust_${LOSS_FUN}" \
    --grid_out_dir "$ROBUST_GRID_OUT_DIR" \
    --grid_noise_methods "none" \
    --loss_fun "$LOSS_FUN" \
    --gce_q 0.7 \
    --bootstrap_beta 0.8
done

echo "Finished at $(date)"
echo "Aggregate JSON: $GRID_JSON_PATH"
