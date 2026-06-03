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
mkdir -p "$START_DIR/logs/1NN_CIFAR10_B10_AUTO_VIZ"

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
LOG_FILE="$START_DIR/logs/1NN_CIFAR10_B10_AUTO_VIZ/${METHOD}_s${SEED}_${TASK_TIME}.out"
exec > "$LOG_FILE" 2>&1

# 5. Environment & Paths
LAB_DIR=/cs/labs/daphna/$USER
PROJECT_DIR=$LAB_DIR/projects/daphna_lab
source $LAB_DIR/envs/venv311/bin/activate
cd $PROJECT_DIR/deep_al/tools

# 6. W&B Grouping Logic
# Distinct COLLECTION for the CIFAR10 viz run; mirrors the CIFAR100 viz
# collection naming so the two are easy to find side-by-side in W&B.
COLLECTION_NAME="FRAMEWORK2_1NN_CIFAR10_B10_JUNE3_AUTO_VIZ"
GROUP_NAME="${METHOD}_NN1_auto_Group"
EXP_NAME="${METHOD}_NN1_auto_viz_s${SEED}_${TASK_TIME}"

echo "Collection: $COLLECTION_NAME"
echo "Group: $GROUP_NAME"
echo "Experiment: $EXP_NAME"
echo "Method: $METHOD | Eval: NN1 (normalized, new default) | Budget: 10 | Seed: $SEED | VIZ: on (tsne)"

# 7. Probcover-only extra flag
# CIFAR10 has NO row in PROBCOVER_AUTO_DELTAS (only CIFAR100 is pinned), so
# 'auto_delta' falls back to the KNEE algorithm with a warning. The [delta]
# banner will say so explicitly. Pass an exact float here if you want to
# bypass that fallback (e.g. --initial_delta 0.5).
EXTRA_FLAGS=()
if [[ "$METHOD" == "probcover" ]]; then
  EXTRA_FLAGS+=(--initial_delta auto_delta)
  echo "ProbCover: --initial_delta auto_delta (expect KNEE fallback; CIFAR10 not in PROBCOVER_AUTO_DELTAS)"
fi

# 8. Run
# --visualize_features enables the per-episode 2D scatter overlay; the t-SNE
# default reads results/feature-space/CIFAR10_tsne_embeddings.npy and the
# matching labels / class_names file. Plots auto-route into
# results/feature-space-visual/$COLLECTION_NAME/ (train_al.py derives this
# from --collection; pass --visualize_output_dir to override).
python3 train_al.py \
  --cfg ../configs/cifar10/al/RESNET18.yaml \
  --al "$METHOD" \
  --exp-name "$EXP_NAME" \
  --group "$GROUP_NAME" \
  --collection "$COLLECTION_NAME" \
  --seed "$SEED" \
  --budget 10 \
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
# CIFAR10 sibling of run_framework2_cifar100_1nn_b100_auto_viz.sh. Same
# protocol (4 AL methods x 5 seeds, NN1 eval, budget = num_classes, 5 AL
# rounds from an empty pool), plus per-episode feature-space PNGs.
#
# After every AL round, train_al.py calls
# deep_al/tools/utils.py:visualize_feature_space, which overlays the current
# lSet (gray X) and the newly-picked activeSet (black X) on the precomputed
# 2D embedding of CIFAR10 (colored by ground-truth class).
#
# Prerequisite (one-time per dataset):
#   python deep_al/tools/visualize_feature_space.py --dataset CIFAR10 --method tsne
# Already present at:
#   results/feature-space/CIFAR10_tsne_embeddings.npy
#   results/feature-space/CIFAR10_labels.npy
#   results/feature-space/CIFAR10_class_names.json
#
# Output PNGs:
#   results/feature-space-visual/${COLLECTION_NAME}/{method}_ep{N}_CIFAR10_tsne_seed{S}_labeled_{n}.png
# 4 methods x 5 seeds x 5 episodes -> 100 unique plots, none clobbered.
# Per-COLLECTION_NAME subfolder keeps this run's plots separate from any
# concurrent .sh run (e.g. the CIFAR100 sibling).
#
# ProbCover delta caveat:
#   CIFAR10 is NOT pinned in PROBCOVER_AUTO_DELTAS (only CIFAR100 is). With
#   --initial_delta auto_delta the [delta] banner will report the table miss
#   and run the KNEE algorithm. To bypass the knee fallback and pin a value,
#   either add a row to PROBCOVER_AUTO_DELTAS in
#   deep_al/pycls/al/delta_defaults.py, or pass --initial_delta <float> here.
#
# Settings:
#   DATASET:        CIFAR10
#   BUDGET/round:   10  (= num_classes)
#   MAX ITER:       5
#   INITIAL LABEL:  empty
#   SEEDS:          1..5
#   VIZ METHOD:     tsne (swap to pca or umap by editing --visualize_method)
# ==============================================================================
