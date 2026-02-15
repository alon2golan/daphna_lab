#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c 8
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/probcover_cifar100_b200_seed0-%j.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/probcover_cifar100_b200_seed0-%j.err
#SBATCH --killable
PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
cd deep_al/tools
python train_al.py \
  --cfg ../configs/cifar100/al/RESNET18.yaml \
  --al prob_cover \
  --exp-name repro_probcover_cifar100_b200_seed0 \
  --initial_size 0 \
  --budget 200 \
  --eval_model_type from_images \
  --kernel_type tophat \
  --initial_delta 0.65 \
  --diff_method prob_cover \
  --sparse_K \
  --matrix_dtype float16 \
  --seed 0
