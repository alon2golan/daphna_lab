#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=48g
#SBATCH -c 8
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-39
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --killable


# 1. Setup Directories
START_DIR=$(pwd)
mkdir -p "$START_DIR/logs/1NN_CIFAR10_Attempt2"

# 2. Method Mapping
# Removing Badge and TypiClust-DP because of lack of implementation / necessary npy files
METHODS=("typiclust" "random" "coreset" "bald" "dbal" "entropy" "margin" "uncertainty")
METHOD_IDX=$((SLURM_ARRAY_TASK_ID / 5))
METHOD=${METHODS[$METHOD_IDX]}

# Map indices 0-4 to exact seeds 1-5
SEED_ID=$((SLURM_ARRAY_TASK_ID % 5))
SEED=$((SEED_ID + 1))
TASK_TIME=$(date +"%H%M") # per task start

# 3. Network Staggering Buffer (Stops jobs from attacking W&B simultaneously)
sleep $((1 + RANDOM % 45))

# 4. Clean Logging
LOG_FILE="$START_DIR/logs/1NN_CIFAR10_Attempt2/${METHOD}_s${SEED}_${TASK_TIME}.out"
exec > "$LOG_FILE" 2>&1

# 5. Environment & Paths
LAB_DIR=/cs/labs/daphna/$USER
PROJECT_DIR=$LAB_DIR/projects/daphna_lab
source $LAB_DIR/envs/venv311/bin/activate
cd $PROJECT_DIR/deep_al/tools

# 6. W&B Grouping Logic
COLLECTION_NAME="FRAMEWORK2_1NN_CIFAR10_B10_Attempt2"
GROUP_NAME="${METHOD}_1NN_Group"
EXP_NAME="${METHOD}_1nn_s${SEED}_${TASK_TIME}"

echo "Collection: $COLLECTION_NAME"
echo "Group: $GROUP_NAME"
echo "Experiment: $EXP_NAME"

# 7. Run
# Targeted the 1NN feature configuration and updated the evaluation type flag
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
  --eval_model_type "from_features"

# ==============================================================================
# RECONSTRUCTING RESULTS: PROTOCOL DATA (based on TypiClust article)
# ==============================================================================

# DATABASE: CIFAR10
# BUDGET: 10 (Equals the number of classes)
# MAX ITERATIONS: 5 (Matches the article's cumulative evaluation step structure)
# INITIAL LABELED POOL: Empty

# PIPELINE MECHANICS PER ROUND:

#   1. SAMPLING:   AL methods pick 10 images.
#                  - TypiClust maps the data into the SimCLR space, groups them into K-Means clusters,
#                    and selects the most typical (dense) center points from clusters lacking labels.
#                  - All other methods also use this exact same pre-trained SimCLR embedding space to extract features
#                    and compute distance or uncertainty vectors for their queries (as defined in the article's setup).

#   2. TRAINING:   No parametric deep network training occurs. The system simply extracts the static SimCLR
#                  embedding vectors of the 10 chosen images and appends them directly into a reference memory matrix.

#   3. EVALUATION: Test set images are converted into SimCLR embeddings. The non-parametric 1-NN predicts
#                  by finding the single closest reference vector stored in the memory matrix lookup table.
#
# SAMPLING METHODS EVALUATED:
#   - Model-Independent (Typiclust - RP, Random, Coreset): Choose points purely based on image geometry or randomness.
#   - Model-Dependent (BALD, DBAL, Entropy, Margin, Uncertainty): Select points by looking at
#                   what the trained model finds confusing.
#
# --- MY CHANGES / COMPLETED BY MYSELF ---

# SEEDS: 1-5 (Reduced from 10 unknown seeds to 5 specific exact seeds)

# OMITTED BASELINES: Removed BADGE and TypiClust-DP due to lack of local implementation / missing prerequisite npy files.
# TypiClust-DP - chooses the images using SCAN - a deep clustering algorithm for both the
#                  self-supervised & clustering tasks (skips SimCLR representation).

# Model-dependent methods cannot start with empty labeled pool since they measure prediction confusion
#                   of the model, so edited them to start with random selection of budget_size (10) samples to initially
#                   populate the reference memory bank.
# ==============================================================================