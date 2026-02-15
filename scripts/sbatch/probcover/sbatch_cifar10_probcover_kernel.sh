#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/cifar10_probcover_kernels-%j.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/cifar10_probcover_kernels-%j.err
#SBATCH --killable
## Use --killable flag to access all GPUs (runs only on daphna lab nodes if removed)
## Active Learning with Kernels on CIFAR-100
## This script runs ProbCover active learning using tophat kernel
## Configuration variables
# Navigate to project directory
PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
cd $PROJECT_DIR
# Activate virtual environment
source ${PROJECT_DIR}/venv/bin/activate
# Print configuration
echo "========================================="
echo "CIFAR-10 ProbCover Active Learning with tophat Kernel"
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
  --cfg ../configs/cifar10/al/RESNET18.yaml \
  --al prob_cover \
  --exp-name repro_probcover_cifar10_b200_seed0 \
  --initial_size 0 \
  --budget 200 \
  --eval_model_type from_images \
  --kernel_type tophat \
  --initial_delta 0.75 \
  --diff_method prob_cover \
  --sparse_K \
  --matrix_dtype float16 \
  --max_iter 37 \
  --seed 0
echo "Job completed!"
