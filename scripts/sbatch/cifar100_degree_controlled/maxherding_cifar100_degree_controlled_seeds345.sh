#!/bin/bash
# Controlled-degree test for CIFAR100 MaxHerding with 1NN.
#
# The goal is to decouple "locality" from "degree". Each config changes delta
# and/or local K so that several different mechanisms induce a similar effective
# graph degree. If accuracy groups by target degree rather than by K or delta,
# that strengthens the claim that graph degree is the main driver.

#SBATCH --job-name=mh_c100_deg_ctl
#SBATCH --mem=64g
#SBATCH -c 4
#SBATCH --time=30:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --array=0-35%4
#SBATCH --output /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/cifar100_degree_controlled/nn1/mh_c100_degree_controlled_%a-%A.txt
#SBATCH --error /cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/cifar100_degree_controlled/nn1/mh_c100_degree_controlled_%a-%A.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/maxherding/cifar100_degree_controlled/nn1"
SEEDS=(3 4 5)

# Format: mode:K:delta:target_band:estimated_effective_degree
# Estimates come from results_summaries/cifar100_degree_evidence_2026_06_27/
# cifar100_delta_k_effective_degree_grid.csv, sampled over CIFAR100 features.
CONFIGS=(
  "global:0:0.65:low:10p4"
  "local:1440:0.60:low:10p2"
  "local:960:0.65:low:13p5"
  "local:5000:0.55:low:9p9"
  "global:0:0.70:optimal:19p0"
  "local:480:0.75:optimal:21p8"
  "local:1440:0.65:optimal:18p3"
  "local:5000:0.60:optimal:19p2"
  "global:0:0.75:high:33p5"
  "local:1440:0.70:high:32p2"
  "local:3500:0.65:high:30p5"
  "local:5000:0.65:high:35p6"
)

CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
SEED=${SEEDS[$SEED_INDEX]}
IFS=":" read -r MODE K DELTA TARGET_BAND DEGREE_TAG <<< "${CONFIGS[$CONFIG_INDEX]}"
DELTA_TAG=${DELTA/./p}
EXP_NAME=maxherding_degree_control_cifar100_seed${SEED}_${MODE}_k${K}_delta${DELTA_TAG}_${TARGET_BAND}_deg${DEGREE_TAG}

mkdir -p "$OUT_DIR"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
cd deep_al/tools

CMD=(
  python train_al.py
  --cfg ../configs/cifar100/al/RESNET18.yaml
  --al max_herding
  --diff_method max_herding
  --kernel_type rbf
  --rbf_cutoff_mode probcover
  --initial_delta "${DELTA}"
  --initial_sigma 1.0
  --sparse_K
  --matrix_dtype float16
  --exp-name "${EXP_NAME}"
  --initial_size 0
  --budget 100
  --max_iter 9
  --seed "${SEED}"
  --eval_model_type NN1
)

if [[ "${MODE}" == "local" ]]; then
  CMD+=(--local_sigma --local_sigma_k "${K}")
fi

"${CMD[@]}"
