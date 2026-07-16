#!/bin/bash
#SBATCH --array=0-2%3
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --gres=gg:g0:1,vmem:20g
#SBATCH --mail-user=alon2golan@gmail.com
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=/cs/labs/daphna/alongol/projects/daphna_lab/logs/sweep/misp_parity_%A_%a.out

# Parity check for the BAYES_MISP port (2026-07-15): same protocol as the runs
# in itai_repro_bayes_al_2025 (Itai's repo, jobs 31077866_0 / 31078001_0):
# CIFAR100 simclr, from_mlp_sklearn eval, tophat delta=0.75, seed 1, 100->1000.
# Arm 2 adds bayes_additive under the SAME eval for the additive-vs-misp gap.
# Expected from Itai's repo: probcover 45.8@1000, bayes_misp(a.5) 45.2@1000.

COMMON="--cfg deep_al/configs/cifar100/al/BAYESCOV_n00.yaml --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --seed 1 --track_progress --tag noise=0.0 --sweep-name bayes_misp_parity__20260715"

declare -A JOBS
JOBS[0]="python deep_al/tools/train_al.py $COMMON --al probcover --exp-name parity_probcover_mlp_seed1"
JOBS[1]="python deep_al/tools/train_al.py $COMMON --al bayes_misp --diff_method max --alpha 0.5 --sparse_K --exp-name parity_bayesmisp_a05_mlp_seed1"
JOBS[2]="python deep_al/tools/train_al.py $COMMON --al bayes_additive --exp-name parity_bayesadditive_mlp_seed1"

CMD=${JOBS[$SLURM_ARRAY_TASK_ID]}
echo "Task $SLURM_ARRAY_TASK_ID: $CMD"

cd /cs/labs/daphna/alongol/projects/daphna_lab
source /cs/labs/daphna/alongol/projects/daphna_lab/venv/bin/activate
sleep $(( SLURM_ARRAY_TASK_ID * 30 ))
eval "$CMD"
