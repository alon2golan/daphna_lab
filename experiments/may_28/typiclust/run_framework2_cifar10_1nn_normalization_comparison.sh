#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=48g
#SBATCH -c 8
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-19
#SBATCH --output=slurm_logs/job_%A_%a.out
#SBATCH --error=slurm_logs/job_%A_%a.err
#SBATCH --killable


# 1. Setup Directories
START_DIR=$(pwd)
mkdir -p "$START_DIR/logs/1NN_NORM_COMPARISON_CIFAR10"

# 2. Method x Eval-Type Mapping
# Four (method, eval_type) pairs -> SLURM_ARRAY_TASK_ID / 5 picks the pair,
# SLURM_ARRAY_TASK_ID % 5 picks the seed (1..5). Total tasks = 4 * 5 = 20.
METHODS=("typiclust"      "typiclust"        "random"     "random")
EVAL_TYPES=("NN1_non_normalized" "NN1"   "NN1_non_normalized"   "NN1")
PAIR_IDX=$((SLURM_ARRAY_TASK_ID / 5))
METHOD=${METHODS[$PAIR_IDX]}
EVAL_TYPE=${EVAL_TYPES[$PAIR_IDX]}

# Map indices 0-4 to exact seeds 1-5
SEED_ID=$((SLURM_ARRAY_TASK_ID % 5))
SEED=$((SEED_ID + 1))
TASK_TIME=$(date +"%H%M") # per task start

# 3. Network Staggering Buffer (Stops jobs from attacking W&B simultaneously)
sleep $((1 + RANDOM % 45))

# 4. Clean Logging
LOG_FILE="$START_DIR/logs/1NN_NORM_COMPARISON_CIFAR10/${METHOD}_${EVAL_TYPE}_s${SEED}_${TASK_TIME}.out"
exec > "$LOG_FILE" 2>&1

# 5. Environment & Paths
LAB_DIR=/cs/labs/daphna/$USER
PROJECT_DIR=$LAB_DIR/projects/daphna_lab
source $LAB_DIR/envs/venv311/bin/activate
cd $PROJECT_DIR/deep_al/tools

# 6. W&B Grouping Logic
# Single COLLECTION for all 4 variants -> they show up on the same plot.
# Distinct GROUP per (method, eval_type) -> per-curve mean+/-std aggregation.
COLLECTION_NAME="FRAMEWORK2_1NN_NORM_COMPARISON_CIFAR10_B10"
GROUP_NAME="${METHOD}_${EVAL_TYPE}_Group"
EXP_NAME="${METHOD}_${EVAL_TYPE}_s${SEED}_${TASK_TIME}"

echo "Collection: $COLLECTION_NAME"
echo "Group: $GROUP_NAME"
echo "Experiment: $EXP_NAME"
echo "Method: $METHOD | Eval: $EVAL_TYPE | Seed: $SEED"

# 7. Run
python3 train_al.py \
  --cfg ../configs/cifar10/al/RESNET18.yaml \
  --al "$METHOD" \
  --exp-name "$EXP_NAME" \
  --group "$GROUP_NAME" \
  --collection "$COLLECTION_NAME" \
  --seed "$SEED" \
  --budget 10 \
  --initial_size 0 \
  --max_iter 5 \
  --eval_model_type "$EVAL_TYPE"

# ==============================================================================
# PURPOSE
# ==============================================================================
#
# Compare 1-NN evaluation with and without L2-normalized features, for the two
# AL methods that work under USE_1NN (typiclust and random). All 4 variants
# land in the same W&B collection so the mean+std curves plot on one graph:
#
#   typiclust + NN1_non_normalized   (raw-Euclidean baseline)
#   typiclust + NN1                  (cosine-similarity NN on the unit hypersphere; new default)
#   random    + NN1_non_normalized   (raw-Euclidean baseline)
#   random    + NN1                  (cosine-similarity NN on the unit hypersphere; new default)
#
# Why: SimCLR features live on (approximately) the unit hypersphere, so 1-NN
# under Euclidean distance over UN-normalized features can be dominated by
# magnitude differences rather than direction. Normalizing puts the lookup in
# the cosine-similarity regime the embedding was trained for. This script
# checks whether that change actually moves the numbers, and also serves as a
# smoke test that the post-refactor branch (MODEL.HEAD_TYPE='knn' alias,
# Fix 1 compatibility check, Fix 2 flag split) still runs both AL methods
# end-to-end under the NN1 eval path.
#
# Settings (matched to the framework2 protocol used in the other scripts here):
#   DATASET:        CIFAR10
#   BUDGET:         10  (= number of classes)
#   MAX ITER:       5
#   INITIAL LABEL:  empty
#   SEEDS:          1..5
#
# Only typiclust and random are included because under USE_1NN the other AL
# methods (coreset, bald, dbal, entropy, margin, uncertainty) need a parametric
# classifier and are now rejected upfront by validate_method_eval_compat in
# active_learning.py.
# ==============================================================================
