#!/bin/bash
#SBATCH --mem=40g
#SBATCH -c 8
#SBATCH --time=96:00:00
#SBATCH --gres=gg:g0:1,vmem:40g
#SBATCH --mail-user=tom.novak@mail.huji.ac.il
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=gpu_rerun_failed_92_seed42_%j.log
#SBATCH --error=gpu_rerun_failed_92_seed42_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1

# Re-run only the 92 failed rows in
# original_features_better_aum_and_asym.csv:
#   - 80 LowBudgetAUM runs;
#   - 6 asymmetric-0.3 NPC-MaxHerding crossval/kNN/oracle runs;
#   - 6 asymmetric-0.3 direct cross-validation runs at 800 labels.

REPO_DIR="/cs/labs/daphna/$USER/projects/daphna_lab"
cd "$REPO_DIR"
source "$REPO_DIR/venv/bin/activate"
cd "$REPO_DIR/deep_al/tools"

CFG_PATH="../configs/cifar100/al/RESNET18.yaml"
SEED="${SEED:-42}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
DEBUG_JSON="${DEBUG_JSON:-1}"

INITIAL_SIZE="${INITIAL_SIZE:-100}"
BUDGET_SIZE="${BUDGET_SIZE:-100}"
BASE_AL_METHODS="typiclust,probcover,maxherding,margin,coreset,random"

CIFAR_N_VARIANT="auto"
CIFAR_N_LABELS_PATH="$REPO_DIR/deep_al/tools/data/CIFAR-100_human.pt"

# Netta LowBudgetAUM settings.
NAS_AUM_EPOCHS="${NAS_AUM_EPOCHS:-40}"
NAS_AUM_FAKE_PERCENTILE="${NAS_AUM_FAKE_PERCENTILE:-80}"
NAS_AUM_FAKE_RATIO="${NAS_AUM_FAKE_RATIO:-1.0}"
NAS_AUM_LR="${NAS_AUM_LR:-0.1}"
NAS_AUM_WEIGHT_DECAY="${NAS_AUM_WEIGHT_DECAY:-0.0001}"
NAS_AUM_MOMENTUM="${NAS_AUM_MOMENTUM:-0.9}"
NAS_AUM_BATCH_SIZE="${NAS_AUM_BATCH_SIZE:-64}"
NAS_NOISE_DROPOUT_MIN_FRACTION="${NAS_NOISE_DROPOUT_MIN_FRACTION:-0.1}"

BASE_GRID_OUT_DIR="$REPO_DIR/grid_outputs/rerun_failed_92_after_debug_seed42"
GRID_JSON_PATH="$BASE_GRID_OUT_DIR/aggregate_results_${RUN_TIMESTAMP}.json"
mkdir -p "$BASE_GRID_OUT_DIR"

JSON_ARGS=(--grid_json_path "$GRID_JSON_PATH")
if [[ "$DEBUG_JSON" == "1" || "$DEBUG_JSON" == "true" || "$DEBUG_JSON" == "TRUE" ]]; then
  JSON_ARGS+=(--debug_json)
fi

CIFAR_N_ARGS=()
if [[ -f "$CIFAR_N_LABELS_PATH" ]]; then
  CIFAR_N_ARGS+=(--cifar_n_labels_path "$CIFAR_N_LABELS_PATH")
else
  echo "WARNING: CIFAR_N_LABELS_PATH does not exist: $CIFAR_N_LABELS_PATH"
fi

run_failed_aum_05_budget () {
  local FINAL_SIZE="$1"
  local MAX_ITER="$2"
  local OUT_DIR="$BASE_GRID_OUT_DIR/noise05/${FINAL_SIZE}"
  mkdir -p "$OUT_DIR"

  local COMMON_ARGS=(
    --cfg "$CFG_PATH"
    --grid_noise_types "clean,sym,asym,cifar100n"
    --noise_rate 0.5
    "${CIFAR_N_ARGS[@]}"
    --cifar_n_variant "$CIFAR_N_VARIANT"
    --seed "$SEED"
    --eval_model_type from_features
    --also_eval_knn
    --knn_k 1
    --paper_hparams
    --initial_size "$INITIAL_SIZE"
    --budget "$BUDGET_SIZE"
    --max_iter "$MAX_ITER"
    --nas_aum_epochs "$NAS_AUM_EPOCHS"
    --nas_aum_fake_ratio "$NAS_AUM_FAKE_RATIO"
    --nas_aum_fake_percentile "$NAS_AUM_FAKE_PERCENTILE"
    --nas_aum_lr "$NAS_AUM_LR"
    --nas_aum_weight_decay "$NAS_AUM_WEIGHT_DECAY"
    --nas_aum_momentum "$NAS_AUM_MOMENTUM"
    --nas_aum_nesterov
    --nas_aum_batch_size "$NAS_AUM_BATCH_SIZE"
    "${JSON_ARGS[@]}"
  )

  # 24 direct AUM runs.
  python3 -u run_noise_al_pipeline.py \
    "${COMMON_ARGS[@]}" \
    --grid_al_methods "$BASE_AL_METHODS" \
    --grid_noise_methods "aum" \
    --loss_fun cross_entropy \
    --exp_name "rerun_failed_aum05_direct_${FINAL_SIZE}" \
    --grid_out_dir "$OUT_DIR/direct_aum"

  # 4 NPC-ProbCover-AUM runs.
  python3 -u run_noise_al_pipeline.py \
    "${COMMON_ARGS[@]}" \
    --grid_al_methods "npc_probcover_aum" \
    --grid_noise_methods "none" \
    --loss_fun cross_entropy \
    --nas_noise_dropout \
    --nas_noise_dropout_min_fraction "$NAS_NOISE_DROPOUT_MIN_FRACTION" \
    --exp_name "rerun_failed_aum05_npc_probcover_${FINAL_SIZE}" \
    --grid_out_dir "$OUT_DIR/npc_probcover_aum"

  # 4 NPC-MaxHerding-AUM runs.
  python3 -u run_noise_al_pipeline.py \
    "${COMMON_ARGS[@]}" \
    --grid_al_methods "npc_maxherding_aum" \
    --grid_noise_methods "none" \
    --loss_fun cross_entropy \
    --nas_noise_dropout \
    --nas_noise_dropout_min_fraction "$NAS_NOISE_DROPOUT_MIN_FRACTION" \
    --exp_name "rerun_failed_aum05_npc_maxherding_${FINAL_SIZE}" \
    --grid_out_dir "$OUT_DIR/npc_maxherding_aum"
}

run_failed_asym03_budget () {
  local FINAL_SIZE="$1"
  local MAX_ITER="$2"
  local OUT_DIR="$BASE_GRID_OUT_DIR/asym03/${FINAL_SIZE}"
  mkdir -p "$OUT_DIR"

  local COMMON_ARGS=(
    --cfg "$CFG_PATH"
    --grid_noise_types "asym"
    --noise_rate 0.3
    --seed "$SEED"
    --eval_model_type from_features
    --also_eval_knn
    --knn_k 1
    --paper_hparams
    --initial_size "$INITIAL_SIZE"
    --budget "$BUDGET_SIZE"
    --max_iter "$MAX_ITER"
    --nas_aum_epochs "$NAS_AUM_EPOCHS"
    --nas_aum_fake_ratio "$NAS_AUM_FAKE_RATIO"
    --nas_aum_fake_percentile "$NAS_AUM_FAKE_PERCENTILE"
    --nas_aum_lr "$NAS_AUM_LR"
    --nas_aum_weight_decay "$NAS_AUM_WEIGHT_DECAY"
    --nas_aum_momentum "$NAS_AUM_MOMENTUM"
    --nas_aum_nesterov
    --nas_aum_batch_size "$NAS_AUM_BATCH_SIZE"
    "${JSON_ARGS[@]}"
  )

  # 6 direct AUM runs.
  python3 -u run_noise_al_pipeline.py \
    "${COMMON_ARGS[@]}" \
    --grid_al_methods "$BASE_AL_METHODS" \
    --grid_noise_methods "aum" \
    --loss_fun cross_entropy \
    --exp_name "rerun_failed_asym03_direct_aum_${FINAL_SIZE}" \
    --grid_out_dir "$OUT_DIR/direct_aum"

  # 1 NPC-ProbCover-AUM run.
  python3 -u run_noise_al_pipeline.py \
    "${COMMON_ARGS[@]}" \
    --grid_al_methods "npc_probcover_aum" \
    --grid_noise_methods "none" \
    --loss_fun cross_entropy \
    --nas_noise_dropout \
    --nas_noise_dropout_min_fraction "$NAS_NOISE_DROPOUT_MIN_FRACTION" \
    --exp_name "rerun_failed_asym03_npc_probcover_aum_${FINAL_SIZE}" \
    --grid_out_dir "$OUT_DIR/npc_probcover_aum"

  # All four NPC-MaxHerding variants failed in the asymmetric-0.3 file.
  python3 -u run_noise_al_pipeline.py \
    "${COMMON_ARGS[@]}" \
    --grid_al_methods "npc_maxherding_crossval,npc_maxherding_knn,npc_maxherding_aum,npc_maxherding_oracle" \
    --grid_noise_methods "none" \
    --loss_fun cross_entropy \
    --nas_noise_dropout \
    --nas_noise_dropout_min_fraction "$NAS_NOISE_DROPOUT_MIN_FRACTION" \
    --exp_name "rerun_failed_asym03_npc_maxherding_${FINAL_SIZE}" \
    --grid_out_dir "$OUT_DIR/npc_maxherding"

  if [[ "$FINAL_SIZE" == "800" ]]; then
    # Six direct cross-validation rows failed only at the 800-label budget.
    python3 -u run_noise_al_pipeline.py \
      "${COMMON_ARGS[@]}" \
      --grid_al_methods "$BASE_AL_METHODS" \
      --grid_noise_methods "crossval" \
      --loss_fun cross_entropy \
      --crossval_folds 5 \
      --crossval_max_iter 300 \
      --exp_name "rerun_failed_asym03_crossval_800" \
      --grid_out_dir "$OUT_DIR/direct_crossval"
  fi
}

echo "Re-running 92 failed rows; aggregate JSON: $GRID_JSON_PATH"

# 64 AUM failures from the 0.5 changed-method batch.
run_failed_aum_05_budget 400 3
run_failed_aum_05_budget 800 7

# 11 failures at asymmetric 0.3 / 400 and 17 at asymmetric 0.3 / 800.
run_failed_asym03_budget 400 3
run_failed_asym03_budget 800 7

echo "Finished re-running all 92 failed rows at $(date)"
echo "Aggregate JSON: $GRID_JSON_PATH"
