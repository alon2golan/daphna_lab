#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c 8
#SBATCH --time=18:00:00
#SBATCH --gres=gg:g4:1
#SBATCH --array=0-2
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/cifar100/global_sigma/fixed_delta/maxherding_cifar100_b100_seed%a-%A.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/cifar100/global_sigma/fixed_delta/maxherding_cifar100_b100_seed%a-%A.err
#SBATCH --killable

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"

SEED=${SLURM_ARRAY_TASK_ID}
EXP_NAME=repro_maxherding_cifar100_b100_seed${SEED}

cd deep_al/tools

python train_al.py \
  --cfg ../configs/cifar100/al/RESNET18.yaml \
  --al max_herding \
  --diff_method max_herding \
  --kernel_type rbf \
  --initial_delta 1.0 \
  --initial_sigma 1.0 \
  --exp-name ${EXP_NAME} \
  --initial_size 0 \
  --budget 100 \
  --max_iter 10 \
  --seed ${SEED} \
  --eval_model_type NN1 \
