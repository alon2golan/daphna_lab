#!/bin/bash
#SBATCH --mem=40g
#SBATCH -c 8
#SBATCH --time=96:00:00
#SBATCH --gres=gg:g0:1,vmem:40g
#SBATCH --mail-user=tom.novak@mail.huji.ac.il
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=gpu_grid_netta_aligned_aum_%j.log
#SBATCH --error=gpu_grid_netta_aligned_aum_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1

# Netta-aligned additions; these do not replace the regular 448-run grid.
#
# Protocol:
#   - CIFAR-100, 20% symmetric noise
#   - cold start (initial labeled set is empty)
#   - b = C = 100 labels per AL round
#   - Netta Framework-2 supervised-training profile
#   - LowBudgetAUM filters the training set for every selector
#   - AUM also acts on the query side for the two NPC selectors
#   - final labeled sizes 400 and 800
#   - five seeds
#
# Methods per seed/budget:
#   random, probcover, maxherding,
#   npc_probcover_aum, npc_maxherding_aum
#
# Count: 5 methods x 5 seeds x 2 budgets = 50 runs.

REPO_DIR="/cs/labs/daphna/$USER/projects/daphna_lab"
cd "$REPO_DIR"
source "$REPO_DIR/venv/bin/activate"
cd "$REPO_DIR/deep_al/tools"

CFG_PATH="../configs/cifar100/al/RESNET18.yaml"
GRID_SEEDS="${GRID_SEEDS:-0,1,2,3,4}"
GRID_AL_METHODS="random,probcover,maxherding,npc_probcover_aum,npc_maxherding_aum"
GRID_NOISE_TYPES="sym"
GRID_NOISE_METHODS="aum"
NOISE_RATE="${NETTA_NOISE_RATE:-0.2}"
INITIAL_SIZE=0
BUDGET_SIZE=100
TRAINING_PROFILE="netta_framework2"

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
  TOTAL_PLANNED_RUNS=$((TOTAL_PLANNED_RUNS + 25))
fi
if [[ "$RUN_FINAL_800" == "1" ]]; then
  TOTAL_PLANNED_RUNS=$((TOTAL_PLANNED_RUNS + 25))
fi

NAS_AUM_EPOCHS="${NAS_AUM_EPOCHS:-40}"
NAS_AUM_FAKE_PERCENTILE="${NAS_AUM_FAKE_PERCENTILE:-80}"
NAS_AUM_FAKE_RATIO="${NAS_AUM_FAKE_RATIO:-1.0}"
NAS_AUM_LR="${NAS_AUM_LR:-0.1}"
NAS_AUM_WEIGHT_DECAY="${NAS_AUM_WEIGHT_DECAY:-0.0001}"
NAS_AUM_MOMENTUM="${NAS_AUM_MOMENTUM:-0.9}"
NAS_AUM_BATCH_SIZE="${NAS_AUM_BATCH_SIZE:-64}"
NAS_NOISE_DROPOUT_MIN_FRACTION="${NAS_NOISE_DROPOUT_MIN_FRACTION:-0.1}"

run_budget () {
  local FINAL_SIZE="$1"
  local MAX_ITER="$2"
  local EXP_NAME="netta_aligned_aum_cold_start_framework2_${FINAL_SIZE}"
  local BASE_GRID_OUT_DIR="$REPO_DIR/grid_outputs/$EXP_NAME"
  local GRID_JSON_PATH="$BASE_GRID_OUT_DIR/aggregate_results_${RUN_TIMESTAMP}.json"

  mkdir -p "$BASE_GRID_OUT_DIR"

  local JSON_ARGS=(--grid_json_path "$GRID_JSON_PATH")
  if [[ "$DEBUG_JSON" == "1" || "$DEBUG_JSON" == "true" || "$DEBUG_JSON" == "TRUE" ]]; then
    JSON_ARGS+=(--debug_json)
  fi

  IFS=',' read -ra SEEDS <<< "$GRID_SEEDS"
  for SEED in "${SEEDS[@]}"; do
    SEED="$(echo "$SEED" | xargs)"
    echo
    echo "Netta-aligned runs: final_size=$FINAL_SIZE seed=$SEED"
    echo "Started at $(date)"

    python3 -u run_noise_al_pipeline.py \
      --cfg "$CFG_PATH" \
      --grid_noise_types "$GRID_NOISE_TYPES" \
      --grid_al_methods "$GRID_AL_METHODS" \
      --grid_noise_methods "$GRID_NOISE_METHODS" \
      --noise_rate "$NOISE_RATE" \
      --seed "$SEED" \
      --eval_model_type from_features \
      --training_profile "$TRAINING_PROFILE" \
      --also_eval_knn \
      --knn_k 1 \
      --paper_hparams \
      --initial_size "$INITIAL_SIZE" \
      --budget "$BUDGET_SIZE" \
      --max_iter "$MAX_ITER" \
      --exp_name "${EXP_NAME}_seed${SEED}" \
      --grid_out_dir "$BASE_GRID_OUT_DIR" \
      --loss_fun cross_entropy \
      --nas_aum_epochs "$NAS_AUM_EPOCHS" \
      --nas_aum_fake_ratio "$NAS_AUM_FAKE_RATIO" \
      --nas_aum_fake_percentile "$NAS_AUM_FAKE_PERCENTILE" \
      --nas_aum_lr "$NAS_AUM_LR" \
      --nas_aum_weight_decay "$NAS_AUM_WEIGHT_DECAY" \
      --nas_aum_momentum "$NAS_AUM_MOMENTUM" \
      --nas_aum_batch_size "$NAS_AUM_BATCH_SIZE" \
      --nas_aum_max_pred_noisy_rate 1.0 \
      --nas_min_labeled_size 0 \
      --nas_noise_dropout \
      --nas_noise_dropout_min_fraction "$NAS_NOISE_DROPOUT_MIN_FRACTION" \
      "${JSON_ARGS[@]}"

    echo "Finished final_size=$FINAL_SIZE seed=$SEED at $(date)"
  done

  echo "Finished Netta-aligned final size $FINAL_SIZE"
  echo "Runs: 25"
  echo "Aggregate JSON: $GRID_JSON_PATH"
}

echo "Starting selected Netta-aligned AUM budgets"
echo "Seeds: $GRID_SEEDS"
echo "Methods: $GRID_AL_METHODS"
echo "Training profile: $TRAINING_PROFILE"
echo "Initial labeled size: $INITIAL_SIZE"
echo "Total runs planned: $TOTAL_PLANNED_RUNS"
echo "400 labels enabled: $RUN_FINAL_400 (25 runs)"
echo "800 labels enabled: $RUN_FINAL_800 (25 runs)"

# Empty initial set + four rounds of 100 = 400.
if [[ "$RUN_FINAL_400" == "1" ]]; then
  run_budget 400 4
fi

# Empty initial set + eight rounds of 100 = 800.
if [[ "$RUN_FINAL_800" == "1" ]]; then
  run_budget 800 8
fi

echo "Finished $TOTAL_PLANNED_RUNS Netta-aligned AUM runs at $(date)"
