#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=32g
#SBATCH -c 4
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --array=0-2
#SBATCH --output=logs/fig4_%A_%a.out

# 1. Create logs folder in the directory where you run sbatch
mkdir -p logs

# 2. Setup Environment
LAB_DIR=/cs/labs/daphna/$USER
PROJECT_DIR=$LAB_DIR/projects/daphna_lab
source $LAB_DIR/envs/venv311/bin/activate

# 3. Logic to split 30 tasks into 3 Methods x 10 Seeds
# Tasks 0-9: TypiClust, 10-19: Random, 20-29: Coreset
METHODS=("typiclust" "random" "coreset")
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
SEED=1  # Keep seed constant for the first test

# 4. Parameters for Figure 4 (Extreme Low Budget)
BUDGET=10        # Budget = Number of Classes (M)
INITIAL_SIZE=0   # Active Initial Selection (Start from zero)
MAX_ITER=5       # Total of 60 labels (0 + 10 + 10 + 10 + 10 + 10)
TIMESTAMP=$(date +"%m%d_%H%M")

# 5. Move to code directory for execution
cd $PROJECT_DIR/deep_al/tools

# 6. Run Experiment
# Note: --eval_model_type is blank to match Figure 4 (Full ResNet-18)
python3 train_al.py \
  --cfg ../configs/cifar10/al/RESNET18.yaml \
  --al "$METHOD" \
  --exp-name "Fig4_${METHOD}_s${SEED}_${TIMESTAMP}" \
  --group "Figure4_CIFAR10${TIMESTAMP}" \
  --budget $BUDGET \
  --initial_size $INITIAL_SIZE \
  --max_iter $MAX_ITER \
  --seed "$SEED" \
  --eval_model_type "from_images"
