#!/bin/bash
#SBATCH --mem=40g
#SBATCH -c 8
#SBATCH --time=96:00:00
#SBATCH --gres=gg:g0:1,vmem:40g
#SBATCH --mail-user=tom.novak@mail.huji.ac.il
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=gpu_rerun_changed_112_seed42_%j.log
#SBATCH --error=gpu_rerun_changed_112_seed42_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1

# ============================================================
# Re-run ONLY experiments affected by the changes made today.
#
# Changed experiment families:
#
# 1) Direct AUM:
#    Old project-specific AUM/relabeling -> corrected LowBudgetAUM filtering.
#    4 noise types x 6 AL methods = 24 runs per final budget.
#
# 2) Bootstrap-soft:
#    Re-run after removing .detach() from the predicted soft target.
#    4 noise types x 6 AL methods = 24 runs per final budget.
#
# 3) NPC-ProbCover + AUM:
#    Re-run with corrected shared LowBudgetAUM detector / uncapped AUM.
#    4 noise types = 4 runs per final budget.
#
# 4) NPC-MaxHerding + AUM:
#    Same LowBudgetAUM changes.
#    4 noise types = 4 runs per final budget.
#
# Total:
#   56 runs for final size 400
#   56 runs for final size 800
#   112 runs total
#
# Unchanged and therefore NOT re-run:
#   - ordinary CE baselines
#   - label_cleaning
#   - crossval cleaning
#   - temporal cleaning
#   - GCE
#   - bootstrap_hard
#   - NPC crossval
#   - NPC kNN
#   - NPC oracle
# ============================================================

REPO_DIR="/cs/labs/daphna/$USER/projects/daphna_lab"
cd "$REPO_DIR"
source "$REPO_DIR/venv/bin/activate"
cd "$REPO_DIR/deep_al/tools"

CFG_PATH="../configs/cifar100/al/RESNET18.yaml"
SEED="${SEED:-42}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
DEBUG_JSON="${DEBUG_JSON:-1}"

GRID_NOISE_TYPES="clean,sym,asym,cifar100n"
BASE_AL_METHODS="typiclust,probcover,maxherding,margin,coreset,random"

NOISE_RATE="${NOISE_RATE:-0.5}"
INITIAL_SIZE="${INITIAL_SIZE:-100}"
BUDGET_SIZE="${BUDGET_SIZE:-100}"

CIFAR_N_VARIANT="auto"
CIFAR_N_LABELS_PATH="$REPO_DIR/deep_al/tools/data/CIFAR-100_human.pt"
SUPERCLASS_GROUPS_PATH=""

# ============================================================
# Corrected LowBudgetAUM parameters
# ============================================================
# Explicitly specified in Netta's LowBudgetAUM:
NAS_AUM_EPOCHS="${NAS_AUM_EPOCHS:-40}"
NAS_AUM_FAKE_PERCENTILE="${NAS_AUM_FAKE_PERCENTILE:-80}"

# Not explicitly specified by Netta; retain the project's assumptions:
NAS_AUM_FAKE_RATIO="${NAS_AUM_FAKE_RATIO:-1.0}"
NAS_AUM_LR="${NAS_AUM_LR:-0.1}"
NAS_AUM_WEIGHT_DECAY="${NAS_AUM_WEIGHT_DECAY:-0.0}"

NAS_MIN_LABELED_SIZE="${NAS_MIN_LABELED_SIZE:-0}"

# Corrected implementation is uncapped. Kept for CLI compatibility.
NAS_AUM_MAX_PRED_NOISY_RATE="${NAS_AUM_MAX_PRED_NOISY_RATE:-1.0}"

# Netta noise dropout is used for NPC query selection only.
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
  echo "CIFAR100N runs may fail."
fi


run_budget () {
  local FINAL_SIZE="$1"
  local MAX_ITER="$2"

  local EXP_NAME="rerun_changed_${FINAL_SIZE}"
  local BASE_GRID_OUT_DIR="$REPO_DIR/grid_outputs/rerun_changed_lowbudgetaum_bootstrapsoft_seed42/${FINAL_SIZE}"
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

    # Shared corrected LowBudgetAUM settings
    --nas_aum_epochs "$NAS_AUM_EPOCHS"
    --nas_aum_fake_ratio "$NAS_AUM_FAKE_RATIO"
    --nas_aum_fake_percentile "$NAS_AUM_FAKE_PERCENTILE"
    --nas_aum_lr "$NAS_AUM_LR"
    --nas_aum_weight_decay "$NAS_AUM_WEIGHT_DECAY"
    --nas_aum_max_pred_noisy_rate "$NAS_AUM_MAX_PRED_NOISY_RATE"
    --nas_min_labeled_size "$NAS_MIN_LABELED_SIZE"

    "${JSON_ARGS[@]}"
  )

  echo
  echo "################################################################"
  echo "# Re-running changed experiments for final size $FINAL_SIZE"
  echo "# MAX_ITER=$MAX_ITER"
  echo "# Planned runs for this budget: 56"
  echo "# Aggregate JSON: $GRID_JSON_PATH"
  echo "################################################################"
  echo

  # ============================================================
  # Block 1: Direct corrected LowBudgetAUM
  # 4 noises x 6 AL methods = 24 runs
  # ============================================================
  AUM_OUT_DIR="$BASE_GRID_OUT_DIR/direct_lowbudget_aum"
  mkdir -p "$AUM_OUT_DIR"

  echo "============================================================"
  echo "[$FINAL_SIZE] Direct LowBudgetAUM: 24 runs"
  echo "Started at $(date)"
  echo "============================================================"

  python3 -u run_noise_al_pipeline.py \
    "${COMMON_ARGS[@]}" \
    --grid_al_methods "$BASE_AL_METHODS" \
    --grid_noise_methods "aum" \
    --exp_name "${EXP_NAME}_direct_lowbudget_aum" \
    --grid_out_dir "$AUM_OUT_DIR" \
    --loss_fun cross_entropy

  echo "Finished [$FINAL_SIZE] Direct LowBudgetAUM at $(date)"


  # ============================================================
  # Block 2: Bootstrap-soft after removing .detach()
  # 4 noises x 6 AL methods = 24 runs
  #
  # IMPORTANT: this assumes losses_noise_methods_v2.py has already
  # been changed from:
  #   model_probs = F.softmax(logits, dim=1).detach()
  # to:
  #   model_probs = F.softmax(logits, dim=1)
  # ============================================================
  BOOTSTRAP_SOFT_OUT_DIR="$BASE_GRID_OUT_DIR/bootstrap_soft_no_detach"
  mkdir -p "$BOOTSTRAP_SOFT_OUT_DIR"

  echo
  echo "============================================================"
  echo "[$FINAL_SIZE] Bootstrap-soft (no detach): 24 runs"
  echo "Started at $(date)"
  echo "============================================================"

  python3 -u run_noise_al_pipeline.py \
    "${COMMON_ARGS[@]}" \
    --grid_al_methods "$BASE_AL_METHODS" \
    --grid_noise_methods "none" \
    --exp_name "${EXP_NAME}_bootstrap_soft_no_detach" \
    --grid_out_dir "$BOOTSTRAP_SOFT_OUT_DIR" \
    --loss_fun bootstrap_soft \
    --bootstrap_beta 0.8

  echo "Finished [$FINAL_SIZE] Bootstrap-soft at $(date)"


  # ============================================================
  # Block 3: NPC-ProbCover + corrected LowBudgetAUM
  # 4 noises x 1 method = 4 runs
  # ============================================================
  NPC_PC_AUM_OUT_DIR="$BASE_GRID_OUT_DIR/npc_probcover_aum"
  mkdir -p "$NPC_PC_AUM_OUT_DIR"

  echo
  echo "============================================================"
  echo "[$FINAL_SIZE] NPC-ProbCover-AUM: 4 runs"
  echo "Started at $(date)"
  echo "============================================================"

  python3 -u run_noise_al_pipeline.py \
    "${COMMON_ARGS[@]}" \
    --grid_al_methods "npc_probcover_aum" \
    --grid_noise_methods "none" \
    --exp_name "${EXP_NAME}_npc_probcover_aum" \
    --grid_out_dir "$NPC_PC_AUM_OUT_DIR" \
    --loss_fun cross_entropy \
    --nas_cv_folds 3 \
    --nas_cv_max_iter 300 \
    --nas_knn_k -1 \
    --nas_noise_dropout \
    --nas_noise_dropout_min_fraction "$NAS_NOISE_DROPOUT_MIN_FRACTION"

  echo "Finished [$FINAL_SIZE] NPC-ProbCover-AUM at $(date)"


  # ============================================================
  # Block 4: NPC-MaxHerding + corrected LowBudgetAUM
  # 4 noises x 1 method = 4 runs
  # ============================================================
  NPC_MH_AUM_OUT_DIR="$BASE_GRID_OUT_DIR/npc_maxherding_aum"
  mkdir -p "$NPC_MH_AUM_OUT_DIR"

  echo
  echo "============================================================"
  echo "[$FINAL_SIZE] NPC-MaxHerding-AUM: 4 runs"
  echo "Started at $(date)"
  echo "============================================================"

  python3 -u run_noise_al_pipeline.py \
    "${COMMON_ARGS[@]}" \
    --grid_al_methods "npc_maxherding_aum" \
    --grid_noise_methods "none" \
    --exp_name "${EXP_NAME}_npc_maxherding_aum" \
    --grid_out_dir "$NPC_MH_AUM_OUT_DIR" \
    --loss_fun cross_entropy \
    --nas_cv_folds 3 \
    --nas_cv_max_iter 300 \
    --nas_knn_k -1 \
    --nas_noise_dropout \
    --nas_noise_dropout_min_fraction "$NAS_NOISE_DROPOUT_MIN_FRACTION"

  echo "Finished [$FINAL_SIZE] NPC-MaxHerding-AUM at $(date)"

  echo
  echo "============================================================"
  echo "Finished changed runs for final size $FINAL_SIZE"
  echo "Runs executed for this budget: 56"
  echo "Aggregate JSON: $GRID_JSON_PATH"
  echo "============================================================"
}


echo "============================================================"
echo "Re-running ONLY experiments affected by today's changes"
echo "Seed: $SEED"
echo "Total planned runs: 112"
echo "  final size 400: 56"
echo "  final size 800: 56"
echo "Started at $(date)"
echo "============================================================"

# 100 initial + 3 query rounds x 100 = 400
run_budget 400 3

# 100 initial + 7 query rounds x 100 = 800
run_budget 800 7

echo
echo "################################################################"
echo "# ALL CHANGED EXPERIMENTS FINISHED"
echo "# Total runs executed: 112"
echo "# Finished at $(date)"
echo "################################################################"
