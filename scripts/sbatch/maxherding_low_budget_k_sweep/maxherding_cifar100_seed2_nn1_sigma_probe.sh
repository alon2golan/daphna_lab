#!/bin/bash
#SBATCH --job-name=mh_c100_s2_nn1_sig
#SBATCH --mem=96g
#SBATCH -c 4
#SBATCH --time=48:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --array=0-12%4
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/sigma_probe_seed2/cifar100/nn1/local_sigma/maxherding_cifar100_seed2_nn1_k%a-%A.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/sigma_probe_seed2/cifar100/nn1/local_sigma/maxherding_cifar100_seed2_nn1_k%a-%A.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/maxherding/sigma_probe_seed2/cifar100/nn1/local_sigma"
SEED=${SEED_OVERRIDE:-2}
K_VALUES=(3 7 60 480 1440 2000 3500 5000 10000 20000 30000 35000 40000)
K=${K_VALUES[$SLURM_ARRAY_TASK_ID]}
EXP_NAME=maxherding_cifar100_seed${SEED}_nn1_local_sigma_k${K}

mkdir -p "$OUT_DIR"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
cd deep_al/tools

python train_al.py \
  --cfg ../configs/cifar100/al/RESNET18.yaml \
  --al max_herding \
  --diff_method max_herding \
  --kernel_type rbf \
  --initial_delta 1.0 \
  --initial_sigma 1.0 \
  --local_sigma \
  --local_sigma_k ${K} \
  --exp-name ${EXP_NAME} \
  --initial_size 0 \
  --budget 200 \
  --max_iter 14 \
  --seed ${SEED} \
  --eval_model_type NN1
