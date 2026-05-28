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
mkdir -p "$START_DIR/logs/1NN_NORM_CIFAR100_B100"

# 2. Method Mapping
# Four AL methods compatible with --eval_model_type NN1_normalized (the NN1
# compat allowlist rejects coreset/bald/dbal/entropy/margin/uncertainty/dcom
# because they need a parametric model for logits / margin / penultimate).
METHODS=("typiclust" "probcover" "maxherding" "random")
METHOD_IDX=$((SLURM_ARRAY_TASK_ID / 5))
METHOD=${METHODS[$METHOD_IDX]}

# Map indices 0-4 to exact seeds 1-5
SEED_ID=$((SLURM_ARRAY_TASK_ID % 5))
SEED=$((SEED_ID + 1))
TASK_TIME=$(date +"%H%M") # per task start

# 3. Network Staggering Buffer (Stops jobs from attacking W&B simultaneously)
sleep $((1 + RANDOM % 45))

# 4. Clean Logging
LOG_FILE="$START_DIR/logs/1NN_NORM_CIFAR100_B100/${METHOD}_s${SEED}_${TASK_TIME}.out"
exec > "$LOG_FILE" 2>&1

# 5. Environment & Paths
LAB_DIR=/cs/labs/daphna/$USER
PROJECT_DIR=$LAB_DIR/projects/daphna_lab
source $LAB_DIR/envs/venv311/bin/activate
cd $PROJECT_DIR/deep_al/tools

# 6. W&B Grouping Logic
# COLLECTION is the per-budget filter; GROUP per method gives one mean+/-std curve per AL.
COLLECTION_NAME="FRAMEWORK2_1NN_NORM_CIFAR100_B100"
GROUP_NAME="${METHOD}_NN1_normalized_Group"
EXP_NAME="${METHOD}_NN1_normalized_s${SEED}_${TASK_TIME}"

echo "Collection: $COLLECTION_NAME"
echo "Group: $GROUP_NAME"
echo "Experiment: $EXP_NAME"
echo "Method: $METHOD | Eval: NN1_normalized | Budget: 100 | Seed: $SEED"

# 7. Run
python3 train_al.py \
  --cfg ../configs/cifar100/al/RESNET18.yaml \
  --al "$METHOD" \
  --exp-name "$EXP_NAME" \
  --group "$GROUP_NAME" \
  --collection "$COLLECTION_NAME" \
  --seed "$SEED" \
  --budget 100 \
  --initial_size 0 \
  --max_iter 5 \
  --eval_model_type "NN1_normalized"

# ==============================================================================
# PURPOSE
# ==============================================================================
#
# CIFAR-100 results under the 1-NN-on-L2-normalized-features evaluator
# (--eval_model_type NN1_normalized), per-round budget = num_classes = 100.
# Comparison across the AL methods that work under USE_1NN.
#
# Settings:
#   DATASET:        CIFAR100
#   BUDGET/round:   100  (= num_classes)
#   MAX ITER:       5    (= number of AL episodes)
#   INITIAL LABEL:  empty
#   SEEDS:          1..5
#
# DCoM is intentionally excluded -- it computes prediction margin via softmax
# over class logits, which requires a parametric classifier; KNeighborsClassifier
# can't produce that. Fix-1 compat validator (active_learning.py) would reject
# the combination at config-load time.
#
# Pair this script with run_framework2_cifar100_1nn_norm_b500.sh (budget 500,
# separate W&B collection + log folder) to see the budget-scaling story.
# ==============================================================================
