#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --array=0-2
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/probcover/cifar10/local_sigma/probcover_cifar10_b200_seed%a-%A.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/probcover/cifar10/local_sigma/probcover_cifar10_b200_seed%a-%A.err
#SBATCH --killable
PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
mkdir -p /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/probcover/cifar10/local_sigma
SEED=${SLURM_ARRAY_TASK_ID}
EXP_NAME=repro_probcover_cifar10_b200_seed${SEED}_local_sigma
cd deep_al/tools
python train_al.py \
  --cfg ../configs/cifar10/al/RESNET18.yaml \
  --al prob_cover \
  --diff_method prob_cover \
  --sparse_K \
  --matrix_dtype float16 \
  --kernel_type tophat \
  --initial_delta 0.75 \
  --local_sigma \
  --local_sigma_k 7 \
  --exp-name ${EXP_NAME} \
  --initial_size 0 \
  --budget 200 \
  --max_iter 37 \
  --seed ${SEED} \
  --eval_model_type from_images
