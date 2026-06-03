#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=48g
#SBATCH -c 8
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-19
#SBATCH --output=slurm_logs/job_%A_%a.out
#SBATCH --error=slurm_logs/job_%A_%a.err
#SBATCH --killable


# 1. Setup Directories
START_DIR=$(pwd)
mkdir -p "$START_DIR/logs/1NN_CIFAR100_B100_AUTO"

# 2. Method Mapping
# Four AL methods compatible with --eval_model_type NN1 (the NN1 compat
# allowlist rejects coreset/bald/dbal/entropy/margin/uncertainty/dcom because
# they need a parametric model for logits / margin / penultimate).
METHODS=("typiclust" "probcover" "maxherding" "random")
METHOD_IDX=$((SLURM_ARRAY_TASK_ID / 5))
METHOD=${METHODS[$METHOD_IDX]}

# Map indices 0-4 to exact seeds 1-5
SEED_ID=$((SLURM_ARRAY_TASK_ID % 5))
SEED=$((SEED_ID + 1))
TASK_TIME=$(date +"%H%M") # per task start

# 3. Network Staggering Buffer (Stops jobs from attacking W&B and accessing data simultaneously)
sleep $((1 + SLURM_ARRAY_TASK_ID * 30))

# 4. Clean Logging
LOG_FILE="$START_DIR/logs/1NN_CIFAR100_B100_AUTO/${METHOD}_s${SEED}_${TASK_TIME}.out"
exec > "$LOG_FILE" 2>&1

# 5. Environment & Paths
LAB_DIR=/cs/labs/daphna/$USER
PROJECT_DIR=$LAB_DIR/projects/daphna_lab
source $LAB_DIR/envs/venv311/bin/activate
cd $PROJECT_DIR/deep_al/tools

# 6. W&B Grouping Logic
# COLLECTION is the per-budget filter; GROUP per method gives one mean+/-std curve per AL.
COLLECTION_NAME="FRAMEWORK2_1NN_CIFAR100_B100_JUNE3_AUTO"
GROUP_NAME="${METHOD}_NN1_auto_Group"
EXP_NAME="${METHOD}_NN1_auto_s${SEED}_${TASK_TIME}"

echo "Collection: $COLLECTION_NAME"
echo "Group: $GROUP_NAME"
echo "Experiment: $EXP_NAME"
echo "Method: $METHOD | Eval: NN1 (normalized, new default) | Budget: 100 | Seed: $SEED"

# 7. Probcover-only extra flag
# For probcover we explicitly pass --initial_delta auto_delta so the [delta]
# log line clearly states the chosen mode (this is the new default anyway).
# Other methods do not consult delta, so the flag is omitted for them.
EXTRA_FLAGS=()
if [[ "$METHOD" == "probcover" ]]; then
  EXTRA_FLAGS+=(--initial_delta auto_delta)
  echo "ProbCover: --initial_delta auto_delta (expect PROBCOVER_AUTO_DELTAS lookup)"
fi

# 8. Run
python3 train_al.py \
  --cfg ../configs/cifar100/al/RESNET18.yaml \
  --al "$METHOD" \
  --exp-name "$EXP_NAME" \
  --group "$GROUP_NAME" \
  --collection "$COLLECTION_NAME" \
  --seed "$SEED" \
  --budget 100 \
  --initial_size 0 \
  --max_iter 5 \
  --eval_model_type "NN1" \
  "${EXTRA_FLAGS[@]}"

# ==============================================================================
# PURPOSE
# ==============================================================================
#
# Re-run of experiments/may_28/1nn_all_algorithms/run_framework2_cifar100_1nn_norm_b100.sh
# under the new code, with two intentional changes:
#
#  1. --eval_model_type "NN1" (was "NN1_normalized"). Under the new alias map
#     NN1 is the L2-normalized 1-NN variant by default, so the runs are
#     identical in semantics to the may_28 series.
#  2. --initial_delta auto_delta is passed explicitly when METHOD=probcover,
#     so the [delta] banner names the mode unambiguously. With the pinned
#     PROBCOVER_AUTO_DELTAS table this resolves to delta=0.65 for
#     (CIFAR100, NN1). The may_28 runs went through the old auto-pick path
#     and got whatever ds_utils.select_delta returned at that point.
#
# Settings:
#   DATASET:        CIFAR100
#   BUDGET/round:   100  (= num_classes)
#   MAX ITER:       5
#   INITIAL LABEL:  empty
#   SEEDS:          1..5
#
# A distinct W&B COLLECTION ("..._JUNE3_AUTO") keeps these runs separate
# from the may_28 collection so the comparison is clear.
# ==============================================================================
