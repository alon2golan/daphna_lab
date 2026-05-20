#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=48g
#SBATCH -c 4
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-39%15
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --killable

# 1. Setup Directories
START_DIR=$(pwd)
mkdir -p "$START_DIR/logs/Attempt5"

# 2. Method Mapping
# Removing Badge and TypiClust-DP because of lack of implementation / necessary npy files
METHODS=("typiclust" "random" "coreset" "bald" "dbal" "entropy" "margin" "uncertainty")
METHOD_IDX=$((SLURM_ARRAY_TASK_ID / 5))
METHOD=${METHODS[$METHOD_IDX]}
SEED=$((SLURM_ARRAY_TASK_ID % 5))
TASK_TIME=$(date +"%H%M") # per task start

# 4. Clean Logging
LOG_FILE="$START_DIR/logs/Attempt5/fig4_${METHOD}_s${SEED}_${TASK_TIME}.out"
exec > "$LOG_FILE" 2>&1

# 5. Environment & Paths
LAB_DIR=/cs/labs/daphna/$USER
PROJECT_DIR=$LAB_DIR/projects/daphna_lab
source $LAB_DIR/envs/venv311/bin/activate
cd $PROJECT_DIR/deep_al/tools

# 6. W&B Grouping Logic
# This is the "Whole Title" you asked for. All 100 runs will share this.
COLLECTION_NAME="Figure4_CIFAR10_B10_Results_Attempt5"

# This groups the 10 seeds of the SAME method together for mean+variance.
# e.g., "Group_typiclust"
GROUP_NAME="${METHOD}_Group"

# This is the individual run name.
# e.g., "typiclust_s0_1430"
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