#!/bin/bash
#SBATCH --array=0-1
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --gres=gg:g0:1,vmem:20g
#SBATCH --killable
#SBATCH --output=/cs/labs/daphna/alongol/projects/daphna_lab/logs/sweep/smoke_mlptorch_rkl_%A_%a.out
# SMOKE TEST (2026-07-16) for the two classifier-comparison code paths:
#   task 0 — from_mlp_torch eval head (torch twin of Itai's sklearn MLP; cold
#            start via the config alias FINE_TUNE=False, adam 1e-3, wd 3e-2)
#   task 1 — TRAIN.SOFT_LOSS=rkl (reverse KL) on the pseudo-all soft-target arm
# 1 episode + tiny epoch counts — correctness only, numbers are meaningless.

declare -A JOBS
JOBS[0]="python deep_al/tools/train_al.py --eval_model_type from_mlp_torch --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 1 --max_epoch 5 --cfg deep_al/configs/cifar100/al/BAYESCOV_n00.yaml --al probcover --seed 1 --exp-name smoke_mlptorch_seed1 --sweep-name SMOKE_mlptorch_rkl__20260716"
JOBS[1]="python deep_al/tools/train_al.py --train_label_mode pseudo --pseudo_scope all --soft_loss rkl --eval_model_type from_features --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 1 --max_epoch 5 --normalize_features --cfg deep_al/configs/cifar100/al/BAYESCOV_n00.yaml --al bayes_product --seed 1 --exp-name smoke_rkl_seed1 --sweep-name SMOKE_mlptorch_rkl__20260716"

CMD=${JOBS[$SLURM_ARRAY_TASK_ID]}
echo "Task $SLURM_ARRAY_TASK_ID: $CMD"
cd /cs/labs/daphna/alongol/projects/daphna_lab
source /cs/labs/daphna/alongol/projects/daphna_lab/venv/bin/activate
eval "$CMD"
