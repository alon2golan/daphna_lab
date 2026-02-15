#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c 8
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/probcover_cifar100_b200_seed%a-%A.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/probcover_cifar100_b200_seed%a-%A.err
#SBATCH --killable
PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
SEED=${SLURM_ARRAY_TASK_ID}
EXP_NAME=repro_probcover_cifar100_b200_seed${SEED}
cd deep_al/tools
python train_al.py \
  --cfg ../configs/cifar100/al/RESNET18.yaml \
  --al prob_cover \
  --exp-name ${EXP_NAME} \
  --initial_size 0 \
  --budget 200 \
  --eval_model_type from_images \
  --kernel_type tophat \
  --initial_delta 0.65 \
  --diff_method prob_cover \
  --sparse_K \
  --matrix_dtype float16 \
  --seed ${SEED}
