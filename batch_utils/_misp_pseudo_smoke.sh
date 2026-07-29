#!/bin/bash
#SBATCH --array=0-1
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --gres=gg:g0:1,vmem:20g
#SBATCH --killable
#SBATCH --mail-user=alon2golan@gmail.com
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=/cs/labs/daphna/alongol/projects/daphna_lab/logs/sweep/misp_pseudo_smoke_%A_%a.out

# SMOKE TEST for the bayes_misp x pseudo wiring (MispPosteriorEngine +
# BayesMispAdapter training-target API, 2026-07-23). One linear + one
# torch-MLP run at budget 100 -> 200 (max_iter 1), noise .2. PASS criterion:
# the log contains "[bayes:pseudo] round ... training on X labeled + Y
# pseudo-labeled points" in BOTH episodes (i.e. no silent observed fallback).

declare -A JOBS
JOBS[0]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_features --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 1 --cfg deep_al/configs/cifar100/al/BAYESCOV_n20.yaml --tag noise=0.2 --train_label_mode pseudo --pseudo_scope all --normalize_features --tag feat_norm=l2 --tag pseudo=all --al bayes_misp --diff_method max --alpha 0.5 --sparse_K --seed 1 --exp-name smoke_misp_pseudo_linear_seed1 --sweep-name MISPPSEUDO_smoke__20260723"
JOBS[1]="python deep_al/tools/train_al.py --track_progress --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 1 --cfg deep_al/configs/cifar100/al/BAYESCOV_n20.yaml --tag noise=0.2 --train_label_mode pseudo --pseudo_scope all --soft_loss softce --eval_model_type from_mlp_torch --normalize_features --tag feat_norm=l2 --tag pseudo=all --tag loss=softce --tag head=mlp_torch --al bayes_misp --diff_method max --alpha 0.5 --sparse_K --seed 1 --exp-name smoke_misp_pseudo_mlptorch_seed1 --sweep-name MISPPSEUDO_smoke__20260723"

CMD=${JOBS[$SLURM_ARRAY_TASK_ID]}
echo "Task $SLURM_ARRAY_TASK_ID: $CMD"
cd /cs/labs/daphna/alongol/projects/daphna_lab
source /cs/labs/daphna/alongol/projects/daphna_lab/venv/bin/activate
eval "$CMD"
