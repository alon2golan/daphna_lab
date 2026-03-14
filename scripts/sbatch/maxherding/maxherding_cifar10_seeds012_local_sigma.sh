#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --array=0-2
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/cifar10/local_sigma/fixed_delta/maxherding_cifar10_b10_seed%a-%A.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/cifar10/local_sigma/fixed_delta/maxherding_cifar10_b10_seed%a-%A.err
#SBATCH --killable

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"

SEED=${SLURM_ARRAY_TASK_ID}
EXP_NAME=repro_maxherding_cifar10_b10_seed${SEED}_local_sigma

cd deep_al/tools

python train_al.py \
  --cfg ../configs/cifar10/al/RESNET18.yaml \
  --al max_herding \
  --diff_method max_herding \
  --kernel_type rbf \
  --initial_delta 1.0 \
  --initial_sigma 1.0 \
  --local_sigma \
  --local_sigma_k 7 \
  --exp-name ${EXP_NAME} \
  --initial_size 0 \
  --budget 10 \
  --max_iter 10 \
  --seed ${SEED} \
  --eval_model_type NN1 \
