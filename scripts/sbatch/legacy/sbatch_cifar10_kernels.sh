#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/cifar10_kernels-%j.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/cifar10_kernels-%j.err
#SBATCH --killable

## Available GPU options based on sinfo:
##- gpu:rtx2080:1,vmem:10g,gg:g0:1  (creek, dumfries - most available)
## - gpu:a5000:1,vmem:24g,gg:g0:1    (binky, drape - more memory)
## - gpu:a10:1,vmem:22g,gg:g0:1      (arion, ampere)
## - gpu:l4:1,vmem:23g,gg:g0:1       (hasufel, incitatus)
## - gpu:rtx6000:1,vmem:24g,gg:g0:1  (firth)
## Use --killable flag to access all GPUs (runs only on daphna lab nodes if removed)

## Active Learning with Kernels on CIFAR-10
## This script runs ProbCover active learning using kernels from kernels.py

## Configuration variables
KERNEL_TYPE=${KERNEL_TYPE:-rbf}  # rbf or tophat
AL_METHOD=${AL_METHOD:-misp}  # probcover, dcom, typiclust_rp, etc.
BUDGET=${BUDGET:-100}
INITIAL_SIZE=${INITIAL_SIZE:-0}
INITIAL_DELTA=${INITIAL_DELTA:-0.75}
MAX_ITER=${MAX_ITER:-10}
SEED=${SEED:-1}
EXP_NAME=${EXP_NAME:-auto}
EVAL_MODEL_TYPE=${EVAL_MODEL_TYPE:-from_features}  # from_images, from_features, or NN1

# Navigate to project directory
PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
cd $PROJECT_DIR

# Activate virtual environment
source ${PROJECT_DIR}/venv/bin/activate

# Print configuration
echo "========================================="
echo "CIFAR-10 Active Learning with Kernels"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Kernel Type: ${KERNEL_TYPE}"
echo "AL Method: ${AL_METHOD}"
echo "Eval Model Type: ${EVAL_MODEL_TYPE}"
echo "Budget: ${BUDGET}"
echo "Initial Size: ${INITIAL_SIZE}"
echo "Initial Delta: ${INITIAL_DELTA}"
echo "Max Iterations: ${MAX_ITER}"
echo "Seed: ${SEED}"
echo "========================================="

# Run active learning
cd deep_al/tools

python train_al.py \
  --cfg ../configs/cifar10/al/RESNET18.yaml \
  --al ${AL_METHOD} \
  --exp-name ${EXP_NAME} \
  --eval_model_type ${EVAL_MODEL_TYPE} \
  --initial_size ${INITIAL_SIZE} \
  --budget ${BUDGET} \
  --initial_delta ${INITIAL_DELTA} \
  --kernel_type ${KERNEL_TYPE} \
  --seed ${SEED} \
  --max_iter ${MAX_ITER}

echo "Job completed!"
