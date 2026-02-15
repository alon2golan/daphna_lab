#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c 8
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:1
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/cifar100_maxherding_b100-%j.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/cifar100_maxherding_b100-%j.err
#SBATCH --killable

## Use --killable flag to access all GPUs (runs only on daphna lab nodes if removed)

## Active Learning with Kernels on CIFAR-100
## This script runs MaxHearding active learning using rbf kernel

## Configuration variables

# Navigate to project directory
PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
cd $PROJECT_DIR

# Activate virtual environment
source ${PROJECT_DIR}/venv/bin/activate

# Print configuration
echo "========================================="
echo "CIFAR-100 MaxHerding Active Learning with rbf Kernel"
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

python ./train_al.py \
  --cfg ../configs/cifar100/al/RESNET18.yaml \
  --al max_herding \
  --exp-name repro_maxherding_cifar100_b100_seed0 \
  --initial_size 0 \
  --budget 100 \
  --eval_model_type NN1 \
  --kernel_type rbf \
  --initial_delta 1.0 \
  --initial_sigma 1.00 \
  --diff_method max_herding \
  --max_iter 10 \
  --seed 0




echo "Job completed!"
