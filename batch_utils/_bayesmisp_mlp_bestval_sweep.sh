#!/bin/bash
#SBATCH --array=0-11%6
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --gres=gg:g0:1,vmem:20g
#SBATCH --mail-user=alon2golan@gmail.com
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=/cs/labs/daphna/alongol/projects/daphna_lab/logs/sweep/bayesmisp_mlpbv2_%A_%a.out

# bayes_misp MLP-head rerun with BEST-VAL selection (2026-07-16): the sklearn
# MLP path now checkpoints on the clean valSet (train_al.py partial_fit loop),
# matching the linear head's selection protocol. Supersedes the MLP rows of
# sweep 31084429 (final-iterate) in ROW_SPEC.

declare -A JOBS
JOBS[0]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n00.yaml --tag noise=0.0 --tag head=mlp_bestval --al bayes_misp --diff_method max --alpha 0.5 --sparse_K --seed 1 --exp-name cfgBAYESCOV_n00_tagnoise=0p0_albayes_misp_mlpbv_seed1 --sweep-name BAYESMISP_mlp_bestval_v2__20260716"
JOBS[1]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n00.yaml --tag noise=0.0 --tag head=mlp_bestval --al bayes_misp --diff_method max --alpha 0.5 --sparse_K --seed 2 --exp-name cfgBAYESCOV_n00_tagnoise=0p0_albayes_misp_mlpbv_seed2 --sweep-name BAYESMISP_mlp_bestval_v2__20260716"
JOBS[2]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n00.yaml --tag noise=0.0 --tag head=mlp_bestval --al bayes_misp --diff_method max --alpha 0.5 --sparse_K --seed 3 --exp-name cfgBAYESCOV_n00_tagnoise=0p0_albayes_misp_mlpbv_seed3 --sweep-name BAYESMISP_mlp_bestval_v2__20260716"
JOBS[3]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n20.yaml --tag noise=0.2 --tag head=mlp_bestval --al bayes_misp --diff_method max --alpha 0.5 --sparse_K --seed 1 --exp-name cfgBAYESCOV_n20_tagnoise=0p2_albayes_misp_mlpbv_seed1 --sweep-name BAYESMISP_mlp_bestval_v2__20260716"
JOBS[4]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n20.yaml --tag noise=0.2 --tag head=mlp_bestval --al bayes_misp --diff_method max --alpha 0.5 --sparse_K --seed 2 --exp-name cfgBAYESCOV_n20_tagnoise=0p2_albayes_misp_mlpbv_seed2 --sweep-name BAYESMISP_mlp_bestval_v2__20260716"
JOBS[5]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n20.yaml --tag noise=0.2 --tag head=mlp_bestval --al bayes_misp --diff_method max --alpha 0.5 --sparse_K --seed 3 --exp-name cfgBAYESCOV_n20_tagnoise=0p2_albayes_misp_mlpbv_seed3 --sweep-name BAYESMISP_mlp_bestval_v2__20260716"
JOBS[6]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n50.yaml --tag noise=0.5 --tag head=mlp_bestval --al bayes_misp --diff_method max --alpha 0.5 --sparse_K --seed 1 --exp-name cfgBAYESCOV_n50_tagnoise=0p5_albayes_misp_mlpbv_seed1 --sweep-name BAYESMISP_mlp_bestval_v2__20260716"
JOBS[7]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n50.yaml --tag noise=0.5 --tag head=mlp_bestval --al bayes_misp --diff_method max --alpha 0.5 --sparse_K --seed 2 --exp-name cfgBAYESCOV_n50_tagnoise=0p5_albayes_misp_mlpbv_seed2 --sweep-name BAYESMISP_mlp_bestval_v2__20260716"
JOBS[8]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n50.yaml --tag noise=0.5 --tag head=mlp_bestval --al bayes_misp --diff_method max --alpha 0.5 --sparse_K --seed 3 --exp-name cfgBAYESCOV_n50_tagnoise=0p5_albayes_misp_mlpbv_seed3 --sweep-name BAYESMISP_mlp_bestval_v2__20260716"
JOBS[9]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n80.yaml --tag noise=0.8 --tag head=mlp_bestval --al bayes_misp --diff_method max --alpha 0.5 --sparse_K --seed 1 --exp-name cfgBAYESCOV_n80_tagnoise=0p8_albayes_misp_mlpbv_seed1 --sweep-name BAYESMISP_mlp_bestval_v2__20260716"
JOBS[10]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n80.yaml --tag noise=0.8 --tag head=mlp_bestval --al bayes_misp --diff_method max --alpha 0.5 --sparse_K --seed 2 --exp-name cfgBAYESCOV_n80_tagnoise=0p8_albayes_misp_mlpbv_seed2 --sweep-name BAYESMISP_mlp_bestval_v2__20260716"
JOBS[11]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n80.yaml --tag noise=0.8 --tag head=mlp_bestval --al bayes_misp --diff_method max --alpha 0.5 --sparse_K --seed 3 --exp-name cfgBAYESCOV_n80_tagnoise=0p8_albayes_misp_mlpbv_seed3 --sweep-name BAYESMISP_mlp_bestval_v2__20260716"

CMD=${JOBS[$SLURM_ARRAY_TASK_ID]}
echo "Task $SLURM_ARRAY_TASK_ID: $CMD"

cd /cs/labs/daphna/alongol/projects/daphna_lab
source /cs/labs/daphna/alongol/projects/daphna_lab/venv/bin/activate
sleep $(( (SLURM_ARRAY_TASK_ID % 6) * 20 ))
eval "$CMD"
