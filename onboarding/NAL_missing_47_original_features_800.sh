#!/bin/bash
#SBATCH --mem=40g
#SBATCH -c 8
#SBATCH --time=96:00:00
#SBATCH --gres=gg:g0:1,vmem:40g
#SBATCH --mail-user=tom.novak@mail.huji.ac.il
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=gpu_missing_47_original_features_800_%j.log
#SBATCH --error=gpu_missing_47_original_features_800_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_DIR="/cs/labs/daphna/$USER/projects/daphna_lab"
cd "$REPO_DIR"
source "$REPO_DIR/venv/bin/activate"
cd "$REPO_DIR/deep_al/tools"

CFG_PATH="../configs/cifar100/al/RESNET18.yaml"
SEED=42

EXP_NAME="missing_47_original_features_800"
BASE_GRID_OUT_DIR="$REPO_DIR/grid_outputs/missing_47_original_features_800"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
DEBUG_JSON="${DEBUG_JSON:-1}"
GRID_JSON_PATH="${GRID_JSON_PATH:-$BASE_GRID_OUT_DIR/aggregate_results_${RUN_TIMESTAMP}.json}"
mkdir -p "$BASE_GRID_OUT_DIR"

# Original/default features are used intentionally.
# Do not add --features_tag.

NOISE_RATE=0.5
INITIAL_SIZE=100
BUDGET_SIZE=100
MAX_ITER=7

BASE_AL_METHODS="typiclust,probcover,maxherding,margin,coreset,random"
NPC_AL_METHODS="npc_probcover_crossval,npc_probcover_knn,npc_probcover_aum,npc_probcover_oracle"

CIFAR_N_VARIANT="auto"
CIFAR_N_LABELS_PATH="$REPO_DIR/deep_al/tools/data/CIFAR-100_human.pt"
SUPERCLASS_GROUPS_PATH=""

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
  echo "CIFAR100N runs may fail."
fi

JSON_ARGS=(--grid_json_path "$GRID_JSON_PATH")
if [[ "$DEBUG_JSON" == "1" || "$DEBUG_JSON" == "true" || "$DEBUG_JSON" == "TRUE" ]]; then
  JSON_ARGS+=(--debug_json)
fi

COMMON_ARGS=(
  --cfg "$CFG_PATH"
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

echo "============================================================"
echo "Running the 47 missing original-feature experiments"
echo "Final labeled size: $((INITIAL_SIZE + BUDGET_SIZE * MAX_ITER))"
echo "Aggregate JSON: $GRID_JSON_PATH"
echo "Started at $(date)"
echo "============================================================"

# ============================================================
# Block 1:
# Missing bootstrap_soft run:
#   asym x random = 1 run
# ============================================================
BOOTSTRAP_SOFT_OUT_DIR="$BASE_GRID_OUT_DIR/bootstrap_soft"
mkdir -p "$BOOTSTRAP_SOFT_OUT_DIR"

echo
echo "============================================================"
echo "Block 1: bootstrap_soft, asym + random, 1 run"
echo "Started at $(date)"
echo "============================================================"

python3 -u run_noise_al_pipeline.py \
  "${COMMON_ARGS[@]}" \
  --grid_noise_types "asym" \
  --grid_al_methods "random" \
  --grid_noise_methods "none" \
  --exp_name "${EXP_NAME}_bootstrap_soft_asym_random" \
  --grid_out_dir "$BOOTSTRAP_SOFT_OUT_DIR" \
  --loss_fun bootstrap_soft \
  --gce_q 0.7 \
  --bootstrap_beta 0.8

echo "Finished Block 1 at $(date)"

# ============================================================
# Block 2:
# Missing bootstrap_soft runs:
#   cifar100n x 6 standard AL methods = 6 runs
# ============================================================
echo
echo "============================================================"
echo "Block 2: bootstrap_soft, CIFAR100N, 6 runs"
echo "Started at $(date)"
echo "============================================================"

python3 -u run_noise_al_pipeline.py \
  "${COMMON_ARGS[@]}" \
  --grid_noise_types "cifar100n" \
  --grid_al_methods "$BASE_AL_METHODS" \
  --grid_noise_methods "none" \
  --exp_name "${EXP_NAME}_bootstrap_soft_cifar100n" \
  --grid_out_dir "$BOOTSTRAP_SOFT_OUT_DIR" \
  --loss_fun bootstrap_soft \
  --gce_q 0.7 \
  --bootstrap_beta 0.8

echo "Finished Block 2 at $(date)"

# ============================================================
# Block 3:
# All missing bootstrap_hard runs:
#   4 noise types x 6 standard AL methods = 24 runs
# ============================================================
BOOTSTRAP_HARD_OUT_DIR="$BASE_GRID_OUT_DIR/bootstrap_hard"
mkdir -p "$BOOTSTRAP_HARD_OUT_DIR"

echo
echo "============================================================"
echo "Block 3: bootstrap_hard, 24 runs"
echo "Started at $(date)"
echo "============================================================"

python3 -u run_noise_al_pipeline.py \
  "${COMMON_ARGS[@]}" \
  --grid_noise_types "clean,sym,asym,cifar100n" \
  --grid_al_methods "$BASE_AL_METHODS" \
  --grid_noise_methods "none" \
  --exp_name "${EXP_NAME}_bootstrap_hard" \
  --grid_out_dir "$BOOTSTRAP_HARD_OUT_DIR" \
  --loss_fun bootstrap_hard \
  --gce_q 0.7 \
  --bootstrap_beta 0.8

echo "Finished Block 3 at $(date)"

# ============================================================
# Block 4:
# All missing NPC-ProbCover runs:
#   4 noise types x 4 NPC methods = 16 runs
# ============================================================
NPC_OUT_DIR="$BASE_GRID_OUT_DIR/npc_probcover"
mkdir -p "$NPC_OUT_DIR"

echo
echo "============================================================"
echo "Block 4: NPC-ProbCover, 16 runs"
echo "Started at $(date)"
echo "============================================================"

python3 -u run_noise_al_pipeline.py \
  "${COMMON_ARGS[@]}" \
  --grid_noise_types "clean,sym,asym,cifar100n" \
  --grid_al_methods "$NPC_AL_METHODS" \
  --grid_noise_methods "none" \
  --exp_name "${EXP_NAME}_npc_probcover" \
  --grid_out_dir "$NPC_OUT_DIR" \
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
echo "Finished all 47 runs at $(date)"
echo "Aggregate JSON: $GRID_JSON_PATH"
echo "============================================================"
