#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding_cifar10_b10_seed0-%j.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding_cifar10_b10_seed0-%j.err
#SBATCH --killable

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"

cd deep_al/tools

python train_al.py \
  --cfg ../configs/cifar10/al/RESNET18.yaml \
  --al max_herding \
  --diff_method max_herding \
  --kernel_type rbf \
  --initial_delta 1.0 \
  --initial_sigma 1.0 \
  --exp-name repro_maxherding_cifar10_b10_seed0 \
  --initial_size 0 \
  --budget 10 \
  --max_iter 10 \
  --seed 0 \
  --eval_model_type NN1 \
