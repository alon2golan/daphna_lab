#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=48g
#SBATCH -c 8
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-54
#SBATCH --output=slurm_logs/job_%A_%a.out
#SBATCH --error=slurm_logs/job_%A_%a.err
#SBATCH --killable


# 1. Setup Directories
START_DIR=$(pwd)
mkdir -p "$START_DIR/logs/PROBCOVER_CIFAR100_B100_DELTA_SWEEP"

# 2. Delta Sweep Mapping
# 11 explicit delta values x 5 seeds = 55 tasks.
# SLURM_ARRAY_TASK_ID / 5 picks the delta, SLURM_ARRAY_TASK_ID % 5 picks the seed.
# DELTA_LABELS are zero-padded to keep alphabetical sort matching numerical sort
# (e.g. d050 < d055 < ... < d100); they end up in W&B group/exp names and on disk.
DELTAS=(       0.50  0.55  0.60  0.65  0.70  0.75  0.80  0.85  0.90  0.95  1.00)
DELTA_LABELS=( d050  d055  d060  d065  d070  d075  d080  d085  d090  d095  d100)

DELTA_IDX=$((SLURM_ARRAY_TASK_ID / 5))
DELTA=${DELTAS[$DELTA_IDX]}
DELTA_LABEL=${DELTA_LABELS[$DELTA_IDX]}

# Map indices 0-4 to exact seeds 1-5
SEED_ID=$((SLURM_ARRAY_TASK_ID % 5))
SEED=$((SEED_ID + 1))
TASK_TIME=$(date +"%H%M") # per task start

# 3. Network Staggering Buffer (Stops jobs from attacking W&B and accessing data simultaneously)
sleep $((1 + SLURM_ARRAY_TASK_ID * 30))

# 4. Clean Logging
LOG_FILE="$START_DIR/logs/PROBCOVER_CIFAR100_B100_DELTA_SWEEP/probcover_${DELTA_LABEL}_s${SEED}_${TASK_TIME}.out"
exec > "$LOG_FILE" 2>&1

# 5. Environment & Paths
LAB_DIR=/cs/labs/daphna/$USER
PROJECT_DIR=$LAB_DIR/projects/daphna_lab
source $LAB_DIR/envs/venv311/bin/activate
cd $PROJECT_DIR/deep_al/tools

# 6. W&B Grouping Logic
# Single COLLECTION for all 11 deltas -> they show up on one plot.
# Distinct GROUP per delta value -> per-curve mean+/-std aggregation across the 5 seeds.
COLLECTION_NAME="PROBCOVER_CIFAR100_B100_DELTA_SWEEP"
GROUP_NAME="probcover_${DELTA_LABEL}_Group"
EXP_NAME="probcover_${DELTA_LABEL}_s${SEED}_${TASK_TIME}"

echo "Collection: $COLLECTION_NAME"
echo "Group: $GROUP_NAME"
echo "Experiment: $EXP_NAME"
echo "AL: probcover | Eval: NN1 | Delta: $DELTA ($DELTA_LABEL) | Budget: 100 | Seed: $SEED"

# 7. Run
# Explicit-float --initial_delta path: resolve_initial_delta will print
# "[delta] ProbCover: using EXPLICIT delta=<value> (from --initial_delta)."
# No table lookup, no knee algorithm. Eval is held fixed at NN1 so the only
# axis of variation across runs is the cover-graph radius.
python3 train_al.py \
  --cfg ../configs/cifar100/al/RESNET18.yaml \
  --al probcover \
  --exp-name "$EXP_NAME" \
  --group "$GROUP_NAME" \
  --collection "$COLLECTION_NAME" \
  --seed "$SEED" \
  --budget 100 \
  --initial_size 0 \
  --max_iter 5 \
  --eval_model_type "NN1" \
  --initial_delta "$DELTA"

# ==============================================================================
# PURPOSE
# ==============================================================================
#
# ProbCover delta SWEEP on CIFAR100, budget 100. Sibling of
# run_probcover_cifar100_b100_delta_comparison.sh: same dataset / budget /
# eval / seeds, but instead of comparing 3 (eval_mode, delta_mode) variants
# this script holds eval fixed at NN1 and sweeps --initial_delta across
# explicit floats 0.50 -> 1.00 in 0.05 steps.
#
# Goal: locate the delta value that maximizes ProbCover test accuracy under
# the NN1 evaluator. The current pinned auto value
# (PROBCOVER_AUTO_DELTAS[('CIFAR100', 'NN1')] = 0.65) sits in the middle of
# the sweep, so this curve also validates -- or motivates updating -- that
# pinned value.
#
# Settings:
#   DATASET:        CIFAR100
#   AL:             probcover
#   BUDGET/round:   100  (= num_classes)
#   MAX ITER:       5
#   INITIAL LABEL:  empty
#   EVAL:           NN1 (normalized 1-NN, the new default)
#   SEEDS:          1..5
#   DELTAS:         {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00}
#
# Tasks: 11 deltas x 5 seeds = 55. Adjust --array if you want a narrower
# range (e.g. --array=0-24 for the first 5 deltas only).
#
# W&B layout: one COLLECTION (PROBCOVER_CIFAR100_B100_DELTA_SWEEP); one GROUP
# per delta value (e.g. probcover_d065_Group). The 5-seed mean+/-std curves
# then plot on one chart, one curve per delta. Grep the per-task .out for
# '[delta]' to confirm each run used the intended value.
# ==============================================================================
