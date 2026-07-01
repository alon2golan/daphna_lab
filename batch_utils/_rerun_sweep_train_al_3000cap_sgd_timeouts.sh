#!/bin/bash
#SBATCH --array=0-1%2
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --gres=gg:g0:1,vmem:20g
#SBATCH --mail-user=alon2golan@gmail.com
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=/cs/labs/daphna/alongol/projects/daphna_lab/logs/sweep/rerun_%A_%a.out

# Rerun of the two tasks (orig array indices 38 & 69) from job 30925479
# (sweep train_al_3000cap_sgd__20260625_101822) that hung at Episode-0 training
# start and were killed at the 24h wall. Configs copied verbatim from
# _generated_sweep_train_al_3000cap_sgd.sh.
declare -A JOBS
JOBS[0]="python /cs/labs/daphna/alongol/projects/daphna_lab/deep_al/tools/train_al.py --initial_delta 0.75 --budget 1000 --eval_model_type from_features --max_iter 2 --cfg deep_al/configs/cifar100/al/RESNET18.yaml --tau 0.1 --nu_prime 0.423177 --al probcover --seed 4 --exp-name probcover_B1000_nucache1000_tau0p1_seed4_sgd --sweep-name train_al_3000cap_sgd__20260625_101822"
JOBS[1]="python /cs/labs/daphna/alongol/projects/daphna_lab/deep_al/tools/train_al.py --initial_delta 0.75 --budget 100 --eval_model_type from_features --max_iter 29 --cfg deep_al/configs/cifar100/al/RESNET18.yaml --tau 0.5 --nu_prime 0.386719 --al random --seed 2 --exp-name random_B100_nucache1000_tau0p5_seed2_sgd --sweep-name train_al_3000cap_sgd__20260625_101822"

CMD=${JOBS[$SLURM_ARRAY_TASK_ID]}
echo "Rerun task $SLURM_ARRAY_TASK_ID: $CMD"

cd /cs/labs/daphna/alongol/projects/daphna_lab
source /cs/labs/daphna/alongol/projects/daphna_lab/venv/bin/activate
eval "$CMD"
