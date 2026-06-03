#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=48g
#SBATCH -c 8
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-14
#SBATCH --output=slurm_logs/job_%A_%a.out
#SBATCH --error=slurm_logs/job_%A_%a.err
#SBATCH --killable


# 1. Setup Directories
START_DIR=$(pwd)
mkdir -p "$START_DIR/logs/PROBCOVER_CIFAR100_B100_DELTA_CMP"

# 2. Variant Mapping
# Three (eval_mode, delta_mode) variants -> SLURM_ARRAY_TASK_ID / 5 picks the
# variant, SLURM_ARRAY_TASK_ID % 5 picks the seed (1..5). Total tasks = 3 * 5 = 15.
#
# Variant 0: NN1 + auto_delta        -> PROBCOVER_AUTO_DELTAS hit, delta=0.65
# Variant 1: NN1 + knee_delta        -> knee algorithm on normalized features
# Variant 2: from_images + auto_delta -> PROBCOVER_AUTO_DELTAS hit, delta=0.75
EVAL_MODES=("NN1"        "NN1"        "from_images")
DELTA_MODES=("auto_delta" "knee_delta" "auto_delta")
VARIANT_LABELS=("NN1_auto" "NN1_knee" "from_images_auto")

VAR_IDX=$((SLURM_ARRAY_TASK_ID / 5))
EVAL_MODE=${EVAL_MODES[$VAR_IDX]}
DELTA_MODE=${DELTA_MODES[$VAR_IDX]}
VARIANT_LABEL=${VARIANT_LABELS[$VAR_IDX]}

# Map indices 0-4 to exact seeds 1-5
SEED_ID=$((SLURM_ARRAY_TASK_ID % 5))
SEED=$((SEED_ID + 1))
TASK_TIME=$(date +"%H%M") # per task start

# 3. Network Staggering Buffer (Stops jobs from attacking W&B and accessing data simultaneously)
sleep $((1 + SLURM_ARRAY_TASK_ID * 30))

# 4. Clean Logging
LOG_FILE="$START_DIR/logs/PROBCOVER_CIFAR100_B100_DELTA_CMP/probcover_${VARIANT_LABEL}_s${SEED}_${TASK_TIME}.out"
exec > "$LOG_FILE" 2>&1

# 5. Environment & Paths
LAB_DIR=/cs/labs/daphna/$USER
PROJECT_DIR=$LAB_DIR/projects/daphna_lab
source $LAB_DIR/envs/venv311/bin/activate
cd $PROJECT_DIR/deep_al/tools

# 6. W&B Grouping Logic
# Single COLLECTION for all 3 variants -> they show up on the same plot.
# Distinct GROUP per variant -> per-curve mean+/-std aggregation.
COLLECTION_NAME="PROBCOVER_CIFAR100_B100_DELTA_CMP"
GROUP_NAME="probcover_${VARIANT_LABEL}_Group"
EXP_NAME="probcover_${VARIANT_LABEL}_s${SEED}_${TASK_TIME}"

echo "Collection: $COLLECTION_NAME"
echo "Group: $GROUP_NAME"
echo "Experiment: $EXP_NAME"
echo "AL: probcover | Eval: $EVAL_MODE | Delta mode: $DELTA_MODE | Budget: 100 | Seed: $SEED"

# 7. Run
python3 train_al.py \
  --cfg ../configs/cifar100/al/RESNET18.yaml \
  --al probcover \
  --exp-name "$EXP_NAME" \
  --group "$GROUP_NAME" \
  --collection "$COLLECTION_NAME" \
  --seed "$SEED" \
  --budget 100 \
  --initial_size 0 \
  --max_iter 5 \
  --eval_model_type "$EVAL_MODE" \
  --initial_delta "$DELTA_MODE"

# ==============================================================================
# PURPOSE
# ==============================================================================
#
# ProbCover delta-mode comparison on CIFAR100, budget 100. Three variants land
# in one W&B collection so the mean+std curves plot together:
#
#   NN1 + auto_delta         -> PROBCOVER_AUTO_DELTAS hit, expected delta=0.65
#                               (1-NN on L2-normalized SimCLR features).
#   NN1 + knee_delta         -> runs ds_utils.select_delta (KMeans pseudo-labels,
#                               purity-curve knee). Same eval model as variant 0;
#                               only the delta picker differs.
#   from_images + auto_delta -> PROBCOVER_AUTO_DELTAS hit, expected delta=0.75
#                               (full ResNet retrained from raw images).
#
# Why: the auto_delta keyword now resolves to a pinned per-(dataset, eval_mode)
# value (0.65 / 0.75 for CIFAR100). This script measures how much the pinned
# value differs from what the knee algorithm picks under the same eval model
# (variants 0 vs 1) and how the entire eval pipeline shifts the result
# (variants 0 vs 2). Both differences directly affect ProbCover's coverage
# graph and the resulting test accuracy curve.
#
# Settings:
#   DATASET:        CIFAR100
#   AL:             probcover
#   BUDGET/round:   100  (= num_classes)
#   MAX ITER:       5
#   INITIAL LABEL:  empty
#   SEEDS:          1..5
#
# All three variants log a single [delta] banner naming the chosen mode and
# the resolved delta value; grep the output for [delta] to confirm.
# ==============================================================================
