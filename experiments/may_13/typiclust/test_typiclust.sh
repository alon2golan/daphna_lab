#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=32g
#SBATCH -c 4
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --array=1-10%10
#SBATCH --output=logs/test_typiclust_%j.out

# 1. Create the logs folder next to this .sh file
mkdir -p logs

# 2. Setup Environment
LAB_DIR=/cs/labs/daphna/$USER/
PROJECT_DIR=$LAB_DIR/projects/daphna_lab
source  $LAB_DIR/envs/venv311/bin/activate

# 3. Go to code, but Slurm will still write logs to the folder we just made
cd $PROJECT_DIR/deep_al/tools

# Project Parameters
SEED=$SLURM_ARRAY_TASK_ID
METHOD="typiclust"
GROUP_NAME="${METHOD}_${TIMESTAMP}"
RUN_NAME="${METHOD}_s${SEED}_${TIMESTAMP}"

# 4. Minimal Test Run
python3 train_al.py \
  --cfg ../configs/cifar10/al/RESNET18.yaml \
  --al $METHOD \
  --exp-name "$RUN_NAME" \
  --group "$GROUP_NAME" \
  --budget 100 \
  --seed "$SEED" \
  --eval_model_type NN1 \
  --max_iter 10 \
  --initial_size 20