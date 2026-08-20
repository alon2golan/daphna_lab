#!/bin/bash
#SBATCH --mem=40g
#SBATCH -c 8
#SBATCH --time=96:00:00
#SBATCH --gres=gg:g0:1,vmem:40g
#SBATCH --mail-user=tom.novak@mail.huji.ac.il
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=gpu_grid_all_400_800_seed42_%j.log
#SBATCH --error=gpu_grid_all_400_800_seed42_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1

# ============================================================
# Complete CIFAR-100 noisy-AL grid for BOTH final budgets:
#   400 labels: initial 100 + 3 x 100 AL queries
#   800 labels: initial 100 + 7 x 100 AL queries
#
# This merges:
#   1) NAL_grid_script_total_framework2_seed42_2.sh
#   2) NAL_grid_script_npc_maxherding_seed42.sh
#
# It intentionally DOES NOT rerun the ordinary MaxHerding baseline
# inside the NPC-MaxHerding block, because that exact baseline is
# already included in the main CE block.
#
# Unique runs:
#   224 for final size 400
#   224 for final size 800
#   448 TOTAL
# ============================================================

REPO_DIR="/cs/labs/daphna/$USER/projects/daphna_lab"
cd "$REPO_DIR"
source "$REPO_DIR/venv/bin/activate"
cd "$REPO_DIR/deep_al/tools"

CFG_PATH="../configs/cifar100/al/RESNET18.yaml"
SEED="${SEED:-42}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
DEBUG_JSON="${DEBUG_JSON:-1}"
RUN_FINAL_400="${RUN_FINAL_400:-1}"
RUN_FINAL_800="${RUN_FINAL_800:-1}"

if [[ "$RUN_FINAL_400" != "0" && "$RUN_FINAL_400" != "1" ]] || \
   [[ "$RUN_FINAL_800" != "0" && "$RUN_FINAL_800" != "1" ]]; then
  echo "RUN_FINAL_400 and RUN_FINAL_800 must be 0 or 1" >&2
  exit 2
fi

TOTAL_PLANNED_RUNS=0
if [[ "$RUN_FINAL_400" == "1" ]]; then
  TOTAL_PLANNED_RUNS=$((TOTAL_PLANNED_RUNS + 224))
fi
if [[ "$RUN_FINAL_800" == "1" ]]; then
  TOTAL_PLANNED_RUNS=$((TOTAL_PLANNED_RUNS + 224))
fi

# ---------------- Shared experiment setup ----------------
GRID_NOISE_TYPES="clean,sym,asym,cifar100n"
BASE_AL_METHODS="typiclust,probcover,maxherding,margin,coreset,random"

NPC_PROBCOVER_METHODS="npc_probcover_crossval,npc_probcover_knn,npc_probcover_aum,npc_probcover_oracle"

# Ordinary maxherding is intentionally omitted here because it is already
# included in BASE_AL_METHODS + noise_method=none + cross_entropy.
NPC_MAXHERDING_METHODS="npc_maxherding_crossval,npc_maxherding_knn,npc_maxherding_aum,npc_maxherding_oracle"

NOISE_RATE="${NOISE_RATE:-0.5}"
INITIAL_SIZE="${INITIAL_SIZE:-100}"
BUDGET_SIZE="${BUDGET_SIZE:-100}"

CIFAR_N_VARIANT="auto"
CIFAR_N_LABELS_PATH="$REPO_DIR/deep_al/tools/data/CIFAR-100_human.pt"
SUPERCLASS_GROUPS_PATH=""

# ---------------- Direct-cleaning hyperparameters ----------------
LABEL_CLEANING_CONF_THRESHOLD="${LABEL_CLEANING_CONF_THRESHOLD:-0.7}"
TEMPORAL_CONF_THRESHOLD="${TEMPORAL_CONF_THRESHOLD:-0.7}"
TEMPORAL_MIN_SEEN="${TEMPORAL_MIN_SEEN:-1}"

# ---------------- LowBudgetAUM / NPC hyperparameters ----------------
# Explicitly reported LowBudgetAUM choices from Netta's work.
NAS_AUM_EPOCHS="${NAS_AUM_EPOCHS:-40}"
NAS_AUM_FAKE_PERCENTILE="${NAS_AUM_FAKE_PERCENTILE:-80}"

# Netta's implementation uses ceil(|L| / C) fake samples. The ratio below is
# a multiplier on that count, so 1.0 reproduces the original count.
NAS_AUM_FAKE_RATIO="${NAS_AUM_FAKE_RATIO:-1.0}"
NAS_AUM_LR="${NAS_AUM_LR:-0.1}"
NAS_AUM_WEIGHT_DECAY="${NAS_AUM_WEIGHT_DECAY:-0.0001}"
NAS_AUM_MOMENTUM="${NAS_AUM_MOMENTUM:-0.9}"
NAS_AUM_BATCH_SIZE="${NAS_AUM_BATCH_SIZE:-64}"

NAS_MIN_LABELED_SIZE="${NAS_MIN_LABELED_SIZE:-0}"

# The corrected LowBudgetAUM implementation is uncapped.
# 1.0 is kept for backward CLI/config compatibility.
NAS_AUM_MAX_PRED_NOISY_RATE="${NAS_AUM_MAX_PRED_NOISY_RATE:-1.0}"

# NPC noise dropout applies only during NPC query selection.
NAS_NOISE_DROPOUT_MIN_FRACTION="${NAS_NOISE_DROPOUT_MIN_FRACTION:-0.1}"

EXTRA_NOISE_ARGS=()
if [[ -n "$SUPERCLASS_GROUPS_PATH" ]]; then
  EXTRA_NOISE_ARGS+=(--superclass_groups_path "$SUPERCLASS_GROUPS_PATH")
fi

EXTRA_CIFAR_N_ARGS=()
if [[ -f "$CIFAR_N_LABELS_PATH" ]]; then
  EXTRA_CIFAR_N_ARGS+=(--cifar_n_labels_path "$CIFAR_N_LABELS_PATH")
else
  echo "WARNING: CIFAR_N_LABELS_PATH does not exist: $CIFAR_N_LABELS_PATH"
  echo "CIFAR100N runs may fail, but the grid will continue until a command fails."
fi

run_budget () {
  local FINAL_SIZE="$1"
  local MAX_ITER="$2"

  local EXP_NAME="total_framework2_${FINAL_SIZE}"
  local BASE_GRID_OUT_DIR="$REPO_DIR/grid_outputs/total_framework2_seed42_${FINAL_SIZE}"
  local GRID_JSON_PATH="$BASE_GRID_OUT_DIR/aggregate_results_${RUN_TIMESTAMP}.json"

  mkdir -p "$BASE_GRID_OUT_DIR"

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

    # Shared LowBudgetAUM configuration.
    # Direct AUM and NPC-AUM now use the same detector implementation.
    --nas_aum_epochs "$NAS_AUM_EPOCHS"
    --nas_aum_fake_ratio "$NAS_AUM_FAKE_RATIO"
    --nas_aum_fake_percentile "$NAS_AUM_FAKE_PERCENTILE"
    --nas_aum_lr "$NAS_AUM_LR"
    --nas_aum_weight_decay "$NAS_AUM_WEIGHT_DECAY"
    --nas_aum_momentum "$NAS_AUM_MOMENTUM"
    --nas_aum_batch_size "$NAS_AUM_BATCH_SIZE"
    --nas_aum_max_pred_noisy_rate "$NAS_AUM_MAX_PRED_NOISY_RATE"
    --nas_min_labeled_size "$NAS_MIN_LABELED_SIZE"

    "${JSON_ARGS[@]}"
  )

  echo
  echo "################################################################"
  echo "# Starting final labeled size = $FINAL_SIZE"
  echo "# MAX_ITER = $MAX_ITER"
  echo "# Output = $BASE_GRID_OUT_DIR"
  echo "# Aggregate JSON = $GRID_JSON_PATH"
  echo "################################################################"
  echo

  # ============================================================
  # Block 1:
  # CE baselines + direct noise-handling methods
  #
  # 4 noise types x 6 AL methods x 5 noise methods = 120 runs
  #
  # NOTE: "aum" now means the corrected LowBudgetAUM filtering
  # implementation, not the old AUM relabeling implementation.
  # ============================================================
  CE_GRID_OUT_DIR="$BASE_GRID_OUT_DIR/block_1_ce_baselines_and_cleaning"
  mkdir -p "$CE_GRID_OUT_DIR"

  echo "============================================================"
  echo "[$FINAL_SIZE] Block 1: CE baselines + noise handling: 120 runs"
  echo "Started at $(date)"
  echo "============================================================"

  python3 -u run_noise_al_pipeline.py \
    "${COMMON_ARGS[@]}" \
    --grid_al_methods "$BASE_AL_METHODS" \
    --grid_noise_methods "none,label_cleaning,crossval,temporal,aum" \
    --exp_name "${EXP_NAME}_block_1_ce_cleaning" \
    --grid_out_dir "$CE_GRID_OUT_DIR" \
    --loss_fun cross_entropy \
    --cleaning_conf_threshold "$LABEL_CLEANING_CONF_THRESHOLD" \
    --cleaning_pred_source checkpoint \
    --crossval_folds 5 \
    --crossval_max_iter 300 \
    --temporal_momentum 0.8 \
    --temporal_min_seen "$TEMPORAL_MIN_SEEN" \
    --temporal_conf_threshold "$TEMPORAL_CONF_THRESHOLD"

  echo "Finished [$FINAL_SIZE] Block 1 at $(date)"

  # ============================================================
  # Block 2:
  # Robust-loss baselines
  #
  # 4 noise types x 6 AL methods x 3 losses = 72 runs
  # ============================================================
  for LOSS_FUN in gce bootstrap_soft bootstrap_hard; do
    ROBUST_GRID_OUT_DIR="$BASE_GRID_OUT_DIR/block_2_robust_${LOSS_FUN}"
    mkdir -p "$ROBUST_GRID_OUT_DIR"

    echo
    echo "============================================================"
    echo "[$FINAL_SIZE] Block 2: robust loss=$LOSS_FUN: 24 runs"
    echo "Started at $(date)"
    echo "============================================================"

    python3 -u run_noise_al_pipeline.py \
      "${COMMON_ARGS[@]}" \
      --grid_al_methods "$BASE_AL_METHODS" \
      --grid_noise_methods "none" \
      --exp_name "${EXP_NAME}_block_2_robust_${LOSS_FUN}" \
      --grid_out_dir "$ROBUST_GRID_OUT_DIR" \
      --loss_fun "$LOSS_FUN" \
      --gce_q 0.7 \
      --bootstrap_beta 0.8

    echo "Finished [$FINAL_SIZE] robust loss=$LOSS_FUN at $(date)"
  done

  # ============================================================
  # Block 3:
  # NPC / NAS ProbCover
  #
  # 4 noise types x 4 NPC ProbCover methods = 16 runs
  # ============================================================
  NPC_PROBCOVER_OUT_DIR="$BASE_GRID_OUT_DIR/block_3_npc_probcover"
  mkdir -p "$NPC_PROBCOVER_OUT_DIR"

  echo
  echo "============================================================"
  echo "[$FINAL_SIZE] Block 3: NPC-ProbCover: 16 runs"
  echo "Started at $(date)"
  echo "============================================================"

  python3 -u run_noise_al_pipeline.py \
    "${COMMON_ARGS[@]}" \
    --grid_al_methods "$NPC_PROBCOVER_METHODS" \
    --grid_noise_methods "none" \
    --exp_name "${EXP_NAME}_block_3_npc_probcover" \
    --grid_out_dir "$NPC_PROBCOVER_OUT_DIR" \
    --loss_fun cross_entropy \
    --nas_cv_folds 3 \
    --nas_cv_max_iter 300 \
    --nas_knn_k -1 \
    --nas_noise_dropout \
    --nas_noise_dropout_min_fraction "$NAS_NOISE_DROPOUT_MIN_FRACTION"

  echo "Finished [$FINAL_SIZE] Block 3 at $(date)"

  # ============================================================
  # Block 4:
  # NPC / NAS MaxHerding
  #
  # 4 noise types x 4 NPC MaxHerding methods = 16 runs
  #
  # The ordinary MaxHerding baseline is NOT repeated here:
  # it already ran in Block 1 under:
  #   al=maxherding, noise_method=none, loss=cross_entropy
  # ============================================================
  NPC_MAXHERDING_OUT_DIR="$BASE_GRID_OUT_DIR/block_4_npc_maxherding"
  mkdir -p "$NPC_MAXHERDING_OUT_DIR"

  echo
  echo "============================================================"
  echo "[$FINAL_SIZE] Block 4: NPC-MaxHerding: 16 runs"
  echo "Started at $(date)"
  echo "============================================================"

  python3 -u run_noise_al_pipeline.py \
    "${COMMON_ARGS[@]}" \
    --grid_al_methods "$NPC_MAXHERDING_METHODS" \
    --grid_noise_methods "none" \
    --exp_name "${EXP_NAME}_block_4_npc_maxherding" \
    --grid_out_dir "$NPC_MAXHERDING_OUT_DIR" \
    --loss_fun cross_entropy \
    --nas_cv_folds 3 \
    --nas_cv_max_iter 300 \
    --nas_knn_k -1 \
    --nas_noise_dropout \
    --nas_noise_dropout_min_fraction "$NAS_NOISE_DROPOUT_MIN_FRACTION"

  echo "Finished [$FINAL_SIZE] Block 4 at $(date)"

  echo
  echo "============================================================"
  echo "Finished final size $FINAL_SIZE at $(date)"
  echo "Runs for this final size: 224"
  echo "Aggregate JSON: $GRID_JSON_PATH"
  echo "============================================================"
}

echo "============================================================"
echo "Starting selected regular experiment budgets"
echo "Seed: $SEED"
echo "Noise types: $GRID_NOISE_TYPES"
echo "Total UNIQUE runs planned: $TOTAL_PLANNED_RUNS"
echo "  400 labels enabled: $RUN_FINAL_400 (224 runs)"
echo "  800 labels enabled: $RUN_FINAL_800 (224 runs)"
echo "Started at $(date)"
echo "============================================================"

# 100 initial labels + 3 * 100 queried = 400 final labels
if [[ "$RUN_FINAL_400" == "1" ]]; then
  run_budget 400 3
fi

# 100 initial labels + 7 * 100 queried = 800 final labels
if [[ "$RUN_FINAL_800" == "1" ]]; then
  run_budget 800 7
fi

echo
echo "################################################################"
echo "# ALL EXPERIMENTS FINISHED"
echo "# Total unique runs executed: $TOTAL_PLANNED_RUNS"
echo "# Finished at $(date)"
echo "################################################################"
