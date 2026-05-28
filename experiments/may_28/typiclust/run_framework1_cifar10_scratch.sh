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
mkdir -p "$START_DIR/logs/Scratch_CIFAR10_Attempt1"

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
LOG_FILE="$START_DIR/logs/Scratch_CIFAR10_Attempt1/${METHOD}_s${SEED}_${TASK_TIME}.out"
exec > "$LOG_FILE" 2>&1

# 5. Environment & Paths
LAB_DIR=/cs/labs/daphna/$USER
PROJECT_DIR=$LAB_DIR/projects/daphna_lab
source $LAB_DIR/envs/venv311/bin/activate
cd $PROJECT_DIR/deep_al/tools

# 6. W&B Grouping Logic
COLLECTION_NAME="FRAMEWORK1_SCRATCH_CIFAR10_B10_Attempt1"
GROUP_NAME="${METHOD}_Scratch_Group"
EXP_NAME="${METHOD}_s${SEED}_${TASK_TIME}"

echo "Collection: $COLLECTION_NAME"
echo "Group: $GROUP_NAME"
echo "Experiment: $EXP_NAME"

# 7. Run
# Added --collection and --seed flags.
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
  --eval_model_type "from_images"


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
#                  and selects the most typical (dense) center points from clusters lacking labels.
#                  - All other methods also use this exact same pre-trained SimCLR embedding space to extract features
#                    and compute distance or uncertainty vectors for their queries.

#   2. TRAINING:   The ResNet18 network is fully reset to random weights. It retrains
#                  from scratch using ONLY the raw pixel images of your gathered labeled set.

#   3. EVALUATION: Trained ResNet18 weights are frozen and tested on raw test-set pixels.
#
# SAMPLING METHODS EVALUATED:
#   - Model-Independent (Typiclust - RP, Random, Coreset): Choose points purely based on image geometry or randomness.
#   - Model-Dependent (BALD, DBAL, Entropy, Margin, Uncertainty): Select points by looking at
#                   what the trained model finds confusing.
#
# --- MY CHANGES / COMPLETED BY MYSELF ---

# SEEDS: 1-5 (Reduced from 10 unknown seeds to 5 specific exact seeds)

# OMITTED BASELINES: Removed BADGE and TypiClust-DP due to lack of local implementation / missing prerequisite npy files.
# TypiClust-DP - chooses the images using SCAN - a deep clsutering algorithm for both the
#                   self-supervised & clustering tasks (skips SimCLR representation).

# Model-dependent methods cannot start with empty labeled pool since they measure prediction confusion
#                   of the model, so edited them to start with random selection of budget_size (10) samples to initially
#                   populate the training space.
# ==============================================================================