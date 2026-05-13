#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=32g
#SBATCH -c 4
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --array=1-5%5
#SBATCH --output=logs/test_typiclust_%A_%a.out  # %A is JobID, %a is Array Index

# 1. Create the logs folder in the directory where YOU ARE standing (the submission dir)
# This ensures the folder exists before Slurm tries to write to it.
mkdir -p logs

# 2. Setup Environment
LAB_DIR=/cs/labs/daphna/$USER/
PROJECT_DIR=$LAB_DIR/projects/daphna_lab
source $LAB_DIR/envs/venv311/bin/activate

# 3. Project Parameters
SEED=$SLURM_ARRAY_TASK_ID
TIMESTAMP=$(date +"%m%d_%H%M")

METHOD="typiclust_rp"
# Using a specific group for this NN1 test to keep it separate from the Fig4 recreation
GROUP_NAME="NN1_Test_${METHOD}_${TIMESTAMP}"
RUN_NAME="${METHOD}_s${SEED}_${TIMESTAMP}"

# 4. Go to code directory
cd $PROJECT_DIR/deep_al/tools

# 5. Test Run (NN1 Mode)
# This stays fast (30 mins is plenty) because eval_model_type is NN1
python3 train_al.py \
  --cfg ../configs/cifar10/al/RESNET18.yaml \
  --al $METHOD \
  --exp-name "$RUN_NAME" \
  --group "$GROUP_NAME" \
  --budget 10 \
  --seed "$SEED" \
  --max_iter 5 \
  --initial_size 0 \
  --eval_model_type NN1