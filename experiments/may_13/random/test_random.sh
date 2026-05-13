#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=32g
#SBATCH -c 4
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --array=1-6%6
#SBATCH --output=random_feat_%A_%a.out

# --- Environment Setup ---
dir=/cs/labs/daphna/$USER/projects/daphna_lab
cd $dir
source /cs/labs/daphna/$USER/envs/venv311/bin/activate
cd $dir/deep_al/tools

# --- Experiment Parameters (Generic) ---
# This timestamp stays the same for all 3 tasks in this array
TIMESTAMP=$(date +"%m%d_%H%M")
SEED=$SLURM_ARRAY_TASK_ID

# Define the logic
METHOD="random"
GROUP_NAME="${METHOD}_feat_${TIMESTAMP}"
RUN_NAME="${METHOD}_s${SEED}_${TIMESTAMP}"

# --- Execute ---
# Note: This assumes you added --group to your train_al.py parser
python3 train_al.py \
  --cfg ../configs/cifar10/al/RESNET18.yaml \
  --al $METHOD \
  --exp-name "$RUN_NAME" \
  --group "$GROUP_NAME" \
  --budget 100 \
  --seed $SEED \
  --eval_model_type NN1 \
  --max_iter 5