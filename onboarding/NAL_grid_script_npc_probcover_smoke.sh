#!/bin/bash
#SBATCH --mem=40g
#SBATCH -c 8
#SBATCH --time=8:00:00
#SBATCH --gres=gg:g0:1,vmem:40g
#SBATCH --mail-user=tom.novak@mail.huji.ac.il
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=gpu_grid_npc_%j.log
#SBATCH --error=gpu_grid_npc_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_DIR="/cs/labs/daphna/$USER/projects/daphna_lab"
cd "$REPO_DIR"
source "$REPO_DIR/venv/bin/activate"
cd "$REPO_DIR/deep_al/tools"

CFG_PATH="../configs/cifar100/al/RESNET18.yaml"
SEED=0
EXP_NAME="npc_probcover_noise_debug_smoke"
BASE_GRID_OUT_DIR="$REPO_DIR/grid_outputs/npc_probcover_noise_debug_smoke"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

DEBUG_JSON="${DEBUG_JSON:-1}"
GRID_JSON_PATH="${GRID_JSON_PATH:-$BASE_GRID_OUT_DIR/aggregate_results_${RUN_TIMESTAMP}.json}"
mkdir -p "$BASE_GRID_OUT_DIR"

GRID_NOISE_TYPES="${GRID_NOISE_TYPES:-sym}"
GRID_AL_METHODS="${GRID_AL_METHODS:-probcover,npc_probcover_oracle}"
GRID_NOISE_METHODS="none"

NOISE_RATE=0.2
INITIAL_SIZE=100
BUDGET_SIZE=100
MAX_ITER=1

NAS_MIN_LABELED_SIZE="${NAS_MIN_LABELED_SIZE:-0}"
NAS_AUM_MAX_PRED_NOISY_RATE="${NAS_AUM_MAX_PRED_NOISY_RATE:-0.5}"

CIFAR_N_VARIANT="auto"
CIFAR_N_LABELS_PATH="$REPO_DIR/deep_al/tools/data/CIFAR-100_human.pt"

EXTRA_CIFAR_N_ARGS=()
if [[ -f "$CIFAR_N_LABELS_PATH" ]]; then
  EXTRA_CIFAR_N_ARGS+=(--cifar_n_labels_path "$CIFAR_N_LABELS_PATH")
else
  echo "WARNING: CIFAR_N_LABELS_PATH does not exist: $CIFAR_N_LABELS_PATH"
fi

JSON_ARGS=(--grid_json_path "$GRID_JSON_PATH")
if [[ "$DEBUG_JSON" == "1" || "$DEBUG_JSON" == "true" || "$DEBUG_JSON" == "TRUE" ]]; then
  JSON_ARGS+=(--debug_json)
fi

echo "GRID_JSON_PATH=$GRID_JSON_PATH"
echo "GRID_NOISE_TYPES=$GRID_NOISE_TYPES"
echo "GRID_AL_METHODS=$GRID_AL_METHODS"
echo "GRID_NOISE_METHODS=$GRID_NOISE_METHODS"
echo "MAX_ITER=$MAX_ITER"
echo "DEBUG_JSON=$DEBUG_JSON"
echo "Started at $(date)"

python3 -u run_noise_al_pipeline.py \
  --cfg "$CFG_PATH" \
  --grid_noise_types "$GRID_NOISE_TYPES" \
  --grid_al_methods "$GRID_AL_METHODS" \
  --grid_noise_methods "$GRID_NOISE_METHODS" \
  --noise_rate "$NOISE_RATE" \
  "${EXTRA_CIFAR_N_ARGS[@]}" \
  --cifar_n_variant "$CIFAR_N_VARIANT" \
  --seed "$SEED" \
  --eval_model_type from_features \
  --also_eval_knn \
  --knn_k 1 \
  --paper_hparams \
  --initial_size "$INITIAL_SIZE" \
  --budget "$BUDGET_SIZE" \
  --max_iter "$MAX_ITER" \
  --exp_name "$EXP_NAME" \
  --grid_out_dir "$BASE_GRID_OUT_DIR" \
  --loss_fun cross_entropy \
  --nas_cv_folds 3 \
  --nas_cv_max_iter 300 \
  --nas_knn_k -1 \
  --nas_aum_epochs 40 \
  --nas_aum_fake_ratio 1.0 \
  --nas_aum_fake_percentile 80 \
  --nas_aum_lr 0.1 \
  --nas_aum_weight_decay 0.0 \
  --nas_noise_dropout \
  --nas_noise_dropout_min_fraction 0.1 \
  --nas_min_labeled_size "$NAS_MIN_LABELED_SIZE" \
  --nas_aum_max_pred_noisy_rate "$NAS_AUM_MAX_PRED_NOISY_RATE" \
  "${JSON_ARGS[@]}"

echo "Finished at $(date)"
echo "Aggregate JSON: $GRID_JSON_PATH"
echo "To inspect noise debug lines, run:"
echo "grep 'NOISE_DEBUG' gpu_grid_npc_${SLURM_JOB_ID}.log"