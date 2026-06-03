#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=48g
#SBATCH -c 8
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-19
#SBATCH --output=slurm_logs/job_%A_%a.out
#SBATCH --error=slurm_logs/job_%A_%a.err
#SBATCH --killable


# 1. Setup Directories
START_DIR=$(pwd)
mkdir -p "$START_DIR/logs/1NN_CIFAR100_B100_VISUALIZE_CHOICES"

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
LOG_FILE="$START_DIR/logs/1NN_CIFAR100_B100_VISUALIZE_CHOICES/${METHOD}_s${SEED}_${TASK_TIME}.out"
exec > "$LOG_FILE" 2>&1

# 5. Environment & Paths
LAB_DIR=/cs/labs/daphna/$USER
PROJECT_DIR=$LAB_DIR/projects/daphna_lab
source $LAB_DIR/envs/venv311/bin/activate
cd $PROJECT_DIR/deep_al/tools

# 6. W&B Grouping Logic
# Distinct COLLECTION from the non-viz JUNE3_AUTO run so the viz collection
# stays cleanly separate (results are semantically identical; this run just
# additionally produces per-episode feature-space PNGs).
COLLECTION_NAME="1NN_CIFAR100_B100_JUNE3_VISUALIZE_CHOICES"
GROUP_NAME="${METHOD}_NN1_auto_Group"
EXP_NAME="${METHOD}_NN1_auto_viz_s${SEED}_${TASK_TIME}"

echo "Collection: $COLLECTION_NAME"
echo "Group: $GROUP_NAME"
echo "Experiment: $EXP_NAME"
echo "Method: $METHOD | Eval: NN1 (normalized, new default) | Budget: 100 | Seed: $SEED | VIZ: on (tsne)"

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
# --visualize_features enables the per-episode 2D scatter overlay; the t-SNE
# default reads results/feature-space/CIFAR100_tsne_embeddings.npy and the
# matching labels / class_names file. Plots auto-route into
# results/feature-space-visual/$COLLECTION_NAME/ (train_al.py derives this
# from --collection; pass --visualize_output_dir to override).
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
  --visualize_features \
  --visualize_method tsne \
  "${EXTRA_FLAGS[@]}"

# ==============================================================================
# PURPOSE
# ==============================================================================
#
# Same protocol as run_framework2_cifar100_1nn_b100_auto.sh, plus per-episode
# feature-space visualizations. After every AL round, train_al.py calls
# deep_al/tools/utils.py:visualize_feature_space, which overlays the current
# lSet (gray X) and the newly-picked activeSet (black X) on the precomputed
# 2D embedding of CIFAR100 (colored by ground-truth class).
#
# Prerequisite (one-time per dataset):
#   python deep_al/tools/visualize_feature_space.py --dataset CIFAR100 --method tsne
# Already present at:
#   results/feature-space/CIFAR100_tsne_embeddings.npy
#   results/feature-space/CIFAR100_labels.npy
#   results/feature-space/CIFAR100_class_names.json
#
# Output PNGs:
#   results/feature-space-visual/${COLLECTION_NAME}/{method}_ep{N}_CIFAR100_tsne_seed{S}_labeled_{n}.png
# With 4 methods x 5 seeds x 5 episodes -> 100 unique plots, none clobbered.
# (The seed token in the filename was added so this script could safely run
# all 5 seeds; see deep_al/tools/utils.py:visualize_feature_space. The
# per-COLLECTION_NAME subfolder isolates plots from other .sh script runs.)
#
# Why a separate _VIZ collection: the numerics are identical to the non-viz
# JUNE3_AUTO collection, but keeping them apart avoids merging averages with
# any earlier non-viz wandb runs that landed in the original collection.
#
# Settings:
#   DATASET:        CIFAR100
#   BUDGET/round:   100  (= num_classes)
#   MAX ITER:       5
#   INITIAL LABEL:  empty
#   SEEDS:          1..5
#   VIZ METHOD:     tsne (swap to pca or umap by editing --visualize_method)
# ==============================================================================
