#!/bin/bash
#SBATCH --array=0-11
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --gres=gg:g0:1,vmem:20g
#SBATCH --killable
#SBATCH --mail-user=alon2golan@gmail.com
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=/cs/labs/daphna/alongol/projects/daphna_lab/logs/sweep/product_mlpbv_%A_%a.out

# bayes_product x OBSERVED labels x sklearn-MLP head (cold + best-val), RAW
# features (2026-07-23): the missing observed-MLP cell of the misp-vs-product
# 2x2x2 grid. Protocol mirrors sweep 31085802 (bayes_misp MLP best-val) exactly
# minus the misp-only flags. 12 runs: noise {0,.2,.5,.8} x seeds {1,2,3}.
# Killable -> no array throttle.

declare -A JOBS
JOBS[0]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n00.yaml --tag noise=0.0 --tag head=mlp_bestval --al bayes_product --seed 1 --exp-name cfgBAYESCOV_n00_tagnoise=0p0_albayes_product_mlpbv_seed1 --sweep-name PRODUCT_mlp_bestval__20260723"
JOBS[1]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n00.yaml --tag noise=0.0 --tag head=mlp_bestval --al bayes_product --seed 2 --exp-name cfgBAYESCOV_n00_tagnoise=0p0_albayes_product_mlpbv_seed2 --sweep-name PRODUCT_mlp_bestval__20260723"
JOBS[2]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n00.yaml --tag noise=0.0 --tag head=mlp_bestval --al bayes_product --seed 3 --exp-name cfgBAYESCOV_n00_tagnoise=0p0_albayes_product_mlpbv_seed3 --sweep-name PRODUCT_mlp_bestval__20260723"
JOBS[3]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n20.yaml --tag noise=0.2 --tag head=mlp_bestval --al bayes_product --seed 1 --exp-name cfgBAYESCOV_n20_tagnoise=0p2_albayes_product_mlpbv_seed1 --sweep-name PRODUCT_mlp_bestval__20260723"
JOBS[4]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n20.yaml --tag noise=0.2 --tag head=mlp_bestval --al bayes_product --seed 2 --exp-name cfgBAYESCOV_n20_tagnoise=0p2_albayes_product_mlpbv_seed2 --sweep-name PRODUCT_mlp_bestval__20260723"
JOBS[5]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n20.yaml --tag noise=0.2 --tag head=mlp_bestval --al bayes_product --seed 3 --exp-name cfgBAYESCOV_n20_tagnoise=0p2_albayes_product_mlpbv_seed3 --sweep-name PRODUCT_mlp_bestval__20260723"
JOBS[6]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n50.yaml --tag noise=0.5 --tag head=mlp_bestval --al bayes_product --seed 1 --exp-name cfgBAYESCOV_n50_tagnoise=0p5_albayes_product_mlpbv_seed1 --sweep-name PRODUCT_mlp_bestval__20260723"
JOBS[7]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n50.yaml --tag noise=0.5 --tag head=mlp_bestval --al bayes_product --seed 2 --exp-name cfgBAYESCOV_n50_tagnoise=0p5_albayes_product_mlpbv_seed2 --sweep-name PRODUCT_mlp_bestval__20260723"
JOBS[8]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n50.yaml --tag noise=0.5 --tag head=mlp_bestval --al bayes_product --seed 3 --exp-name cfgBAYESCOV_n50_tagnoise=0p5_albayes_product_mlpbv_seed3 --sweep-name PRODUCT_mlp_bestval__20260723"
JOBS[9]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n80.yaml --tag noise=0.8 --tag head=mlp_bestval --al bayes_product --seed 1 --exp-name cfgBAYESCOV_n80_tagnoise=0p8_albayes_product_mlpbv_seed1 --sweep-name PRODUCT_mlp_bestval__20260723"
JOBS[10]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n80.yaml --tag noise=0.8 --tag head=mlp_bestval --al bayes_product --seed 2 --exp-name cfgBAYESCOV_n80_tagnoise=0p8_albayes_product_mlpbv_seed2 --sweep-name PRODUCT_mlp_bestval__20260723"
JOBS[11]="python deep_al/tools/train_al.py --track_progress --eval_model_type from_mlp_sklearn --kernel_type tophat --budget 100 --initial_delta 0.75 --max_iter 9 --cfg deep_al/configs/cifar100/al/BAYESCOV_n80.yaml --tag noise=0.8 --tag head=mlp_bestval --al bayes_product --seed 3 --exp-name cfgBAYESCOV_n80_tagnoise=0p8_albayes_product_mlpbv_seed3 --sweep-name PRODUCT_mlp_bestval__20260723"

CMD=${JOBS[$SLURM_ARRAY_TASK_ID]}
echo "Task $SLURM_ARRAY_TASK_ID: $CMD"
cd /cs/labs/daphna/alongol/projects/daphna_lab
source /cs/labs/daphna/alongol/projects/daphna_lab/venv/bin/activate
sleep $(( (SLURM_ARRAY_TASK_ID % 6) * 20 ))
eval "$CMD"
