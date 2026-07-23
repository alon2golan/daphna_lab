#!/bin/bash
#SBATCH --mem=40g
#SBATCH -c 8
#SBATCH --time=96:00:00
#SBATCH --gres=gg:g0:1,vmem:40g
#SBATCH --mail-user=tom.novak@mail.huji.ac.il
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=gpu_grid_total_seed42_%j.log
#SBATCH --error=gpu_grid_total_seed42_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_DIR="/cs/labs/daphna/$USER/projects/daphna_lab"
cd "$REPO_DIR"
source "$REPO_DIR/venv/bin/activate"
cd "$REPO_DIR/deep_al/tools"

CFG_PATH="../configs/cifar100/al/RESNET18.yaml"
SEED=42

EXP_NAME="total_framework2_seed42"
BASE_GRID_OUT_DIR="$REPO_DIR/grid_outputs/total_framework2_seed42"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
DEBUG_JSON="${DEBUG_JSON:-1}"
GRID_JSON_PATH="${GRID_JSON_PATH:-$BASE_GRID_OUT_DIR/aggregate_results_${RUN_TIMESTAMP}.json}"
mkdir -p "$BASE_GRID_OUT_DIR"

# Shared setup
GRID_NOISE_TYPES="clean,sym,asym,cifar100n"
BASE_AL_METHODS="typiclust,probcover,maxherding,margin,coreset,random"
NPC_AL_METHODS="npc_probcover_crossval,npc_probcover_knn,npc_probcover_aum,npc_probcover_oracle"

NOISE_RATE=0.5
INITIAL_SIZE=100
BUDGET_SIZE=100
MAX_ITER=7
CIFAR_N_VARIANT="auto"
CIFAR_N_LABELS_PATH="$REPO_DIR/deep_al/tools/data/CIFAR-100_human.pt"
SUPERCLASS_GROUPS_PATH=""

# Direct-cleaning hyperparameters
LABEL_CLEANING_CONF_THRESHOLD="${LABEL_CLEANING_CONF_THRESHOLD:-0.7}"
TEMPORAL_CONF_THRESHOLD="${TEMPORAL_CONF_THRESHOLD:-0.7}"
TEMPORAL_MIN_SEEN="${TEMPORAL_MIN_SEEN:-1}"
AUM_PERCENTILE="${AUM_PERCENTILE:-20}"
AUM_CONF_THRESHOLD="${AUM_CONF_THRESHOLD:-0.7}"

# NPC hyperparameters
NAS_MIN_LABELED_SIZE="${NAS_MIN_LABELED_SIZE:-0}"
NAS_AUM_MAX_PRED_NOISY_RATE="${NAS_AUM_MAX_PRED_NOISY_RATE:-0.5}"

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
echo "SEED=$SEED"
echo "GRID_NOISE_TYPES=$GRID_NOISE_TYPES"
echo "BASE_AL_METHODS=$BASE_AL_METHODS"
echo "NPC_AL_METHODS=$NPC_AL_METHODS"
echo "BASE_GRID_OUT_DIR=$BASE_GRID_OUT_DIR"
echo "GRID_JSON_PATH=$GRID_JSON_PATH"
echo "DEBUG_JSON=$DEBUG_JSON"
echo "NOISE_RATE=$NOISE_RATE"
echo "INITIAL_SIZE=$INITIAL_SIZE"
echo "BUDGET_SIZE=$BUDGET_SIZE"
echo "MAX_ITER=$MAX_ITER"
echo "CIFAR_N_LABELS_PATH=$CIFAR_N_LABELS_PATH"
echo "Started at $(date)"

# ============================================================
# Block 1 + 2:
# Main AL baselines + direct noise-handling methods.
#
# Runs:
#   4 noise types
#   x 6 AL methods
#   x 5 noise methods: none,label_cleaning,crossval,temporal,aum
#   x cross_entropy
# = 120 runs
#
# The "none" rows are the standard AL baselines.
# The other rows are direct noise-handling / cleaning baselines.
# ============================================================
CE_GRID_OUT_DIR="$BASE_GRID_OUT_DIR/block_1_2_ce_baselines_and_cleaning"
mkdir -p "$CE_GRID_OUT_DIR"

echo
echo "============================================================"
echo "Block 1+2: CE baselines + direct cleaning, 120 runs"
echo "Started at $(date)"
echo "============================================================"

python3 -u run_noise_al_pipeline.py \
  "${COMMON_ARGS[@]}" \
  --grid_al_methods "$BASE_AL_METHODS" \
  --grid_noise_methods "none,label_cleaning,crossval,temporal,aum" \
  --exp_name "${EXP_NAME}_block_1_2_ce_cleaning" \
  --grid_out_dir "$CE_GRID_OUT_DIR" \
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

echo "Finished Block 1+2 at $(date)"

# ============================================================
# Block 3:
# Robust-loss baselines.
#
# Runs:
#   4 noise types
#   x 6 AL methods
#   x 3 robust losses: gce,bootstrap_soft,bootstrap_hard
#   x noise_method none
# = 72 runs
# ============================================================
for LOSS_FUN in gce bootstrap_soft bootstrap_hard; do
  ROBUST_GRID_OUT_DIR="$BASE_GRID_OUT_DIR/block_3_robust_${LOSS_FUN}"
  mkdir -p "$ROBUST_GRID_OUT_DIR"

  echo
  echo "============================================================"
  echo "Block 3: robust loss=${LOSS_FUN}, 24 runs"
  echo "Started at $(date)"
  echo "============================================================"

  python3 -u run_noise_al_pipeline.py \
    "${COMMON_ARGS[@]}" \
    --grid_al_methods "$BASE_AL_METHODS" \
    --grid_noise_methods "none" \
    --exp_name "${EXP_NAME}_block_3_robust_${LOSS_FUN}" \
    --grid_out_dir "$ROBUST_GRID_OUT_DIR" \
    --loss_fun "$LOSS_FUN" \
    --gce_q 0.7 \
    --bootstrap_beta 0.8

  echo "Finished robust loss=${LOSS_FUN} at $(date)"
done

# ============================================================
# Block 4:
# NPC / Netta Framework-2-style methods.
#
# Runs:
#   4 noise types
#   x 4 NPC methods:
#       npc_probcover_crossval,npc_probcover_knn,
#       npc_probcover_aum,npc_probcover_oracle
#   x noise_method none
#   x cross_entropy
# = 16 runs
#
# Regular probcover baseline is not repeated here because it already appears
# in Block 1 with noise_method=none and loss=cross_entropy.
# ============================================================
NPC_GRID_OUT_DIR="$BASE_GRID_OUT_DIR/block_4_npc_framework2"
mkdir -p "$NPC_GRID_OUT_DIR"

echo
echo "============================================================"
echo "Block 4: NPC / Framework 2 methods, 16 runs"
echo "Started at $(date)"
echo "============================================================"

python3 -u run_noise_al_pipeline.py \
  "${COMMON_ARGS[@]}" \
  --grid_al_methods "$NPC_AL_METHODS" \
  --grid_noise_methods "none" \
  --exp_name "${EXP_NAME}_block_4_npc_framework2" \
  --grid_out_dir "$NPC_GRID_OUT_DIR" \
  --loss_fun cross_entropy \
  --nas_cv_folds 3 \
  --nas_cv_max_iter 300 \
  --nas_knn_k -1 \
  --nas_aum_epochs 40 \
  --nas_aum_fake_ratio 1.0 \
  --nas_aum_fake_percentile 80 \
  --nas_aum_lr 0.1 \
  --nas_aum_weight_decay 0.0 \
  --nas_aum_max_pred_noisy_rate "$NAS_AUM_MAX_PRED_NOISY_RATE" \
  --nas_min_labeled_size "$NAS_MIN_LABELED_SIZE" \
  --nas_noise_dropout \
  --nas_noise_dropout_min_fraction 0.1

echo "Finished Block 4 at $(date)"

echo
echo "============================================================"
echo "Finished all blocks at $(date)"
echo "Total planned unique runs: 208"
echo "Aggregate JSON: $GRID_JSON_PATH"
echo "Useful checks:"
echo "  grep 'NOISE_DEBUG' gpu_grid_total_seed42_${SLURM_JOB_ID}.log"
echo "  grep 'status' $GRID_JSON_PATH | head"
echo "============================================================"
