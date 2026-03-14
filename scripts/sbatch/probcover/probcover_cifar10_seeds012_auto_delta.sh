#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --array=0-2
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/probcover/cifar10/global_sigma/auto_delta/probcover_cifar10_b10_seed%a-%A.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/probcover/cifar10/global_sigma/auto_delta/probcover_cifar10_b10_seed%a-%A.err
#SBATCH --killable
PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
SEED=${SLURM_ARRAY_TASK_ID}
EXP_NAME=repro_probcover_cifar10_b10_seed${SEED}_auto_delta
cd deep_al/tools
python train_al.py \
  --cfg ../configs/cifar10/al/RESNET18.yaml \
  --al prob_cover \
  --diff_method prob_cover \
  --sparse_K \
  --matrix_dtype float16 \
  --kernel_type tophat \
  --no_features_normalized \
  --exp-name ${EXP_NAME} \
  --initial_size 0 \
  --budget 10 \
  --max_iter 10 \
  --seed ${SEED} \
  --eval_model_type from_images
