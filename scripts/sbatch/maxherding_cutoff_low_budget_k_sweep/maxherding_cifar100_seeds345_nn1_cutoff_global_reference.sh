#!/bin/bash
#SBATCH --job-name=mh_c100_cut_ref
#SBATCH --mem=64g
#SBATCH -c 6
#SBATCH --time=18:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --array=0-2%1
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/cutoff_seeds345/cifar100/nn1/global_sigma/maxherding_cifar100_cutoff_global_seed%a-%A.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/cutoff_seeds345/cifar100/nn1/global_sigma/maxherding_cifar100_cutoff_global_seed%a-%A.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/maxherding/cutoff_seeds345/cifar100/nn1/global_sigma"
SEEDS=(3 4 5)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
EXP_NAME=maxherding_cutoff_cifar100_seed${SEED}_nn1_global_sigma

mkdir -p "$OUT_DIR"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
cd deep_al/tools

python train_al.py \
  --cfg ../configs/cifar100/al/RESNET18.yaml \
  --al max_herding \
  --diff_method max_herding \
  --kernel_type rbf \
  --rbf_cutoff_mode probcover \
  --initial_delta 0.65 \
  --initial_sigma 1.0 \
  --sparse_K \
  --matrix_dtype float16 \
  --exp-name ${EXP_NAME} \
  --initial_size 0 \
  --budget 100 \
  --max_iter 9 \
  --seed ${SEED} \
  --eval_model_type NN1
