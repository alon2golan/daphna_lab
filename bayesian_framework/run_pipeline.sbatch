#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=32g
#SBATCH -c 4
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:l4:1
#SBATCH --output=bayesian_framework/slurm_logs/pipeline_%j.out
#SBATCH --error=bayesian_framework/slurm_logs/pipeline_%j.err
#SBATCH --killable
#SBATCH --requeue

# Build matrix 1 (distances) + matrix 2 (figures) and render the figure, on GPU.
#
# Usage:
#   sbatch bayesian_framework/run_pipeline.sbatch <EMB_DIR> <DELTA> [K] [METHOD] [SPLIT]
#
# Positional args:
#   1) EMB_DIR   embeddings dir with <split>_features_normalized.npy
#   2) DELTA     tophat cutoff (also used as the neighborhood scale)
#   3) K         neighbors to store in matrix 1 (default 512; keep above the max
#                neighborhood at DELTA so it never binds)
#   4) METHOD    tophat | gaussian | laplacian | cosine (default tophat)
#   5) SPLIT     train | test (default train)
#
# Example (the cifar100 run):
#   sbatch bayesian_framework/run_pipeline.sbatch \
#       solo-learn/embeddings/simclr/cifar100/1000ep 0.65 512

set -e

EMB_DIR=${1:?usage: sbatch run_pipeline.sbatch EMB_DIR DELTA [K] [METHOD] [SPLIT]}
DELTA=${2:?usage: sbatch run_pipeline.sbatch EMB_DIR DELTA [K] [METHOD] [SPLIT]}
K=${3:-512}
METHOD=${4:-tophat}
SPLIT=${5:-train}

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
cd "$ROOT"
# use the verified environment (torch + cuda); fall back to venv if absent
source "$ROOT/.venv_clean/bin/activate" 2>/dev/null || source "$ROOT/venv/bin/activate"
mkdir -p bayesian_framework/slurm_logs

# tag = <method>_<dataset>_<epochs> from the last 3 parts of EMB_DIR, matching
# what distance.py names its output file
EMB_DIR=${EMB_DIR%/}
TAG=$(basename "$(dirname "$(dirname "$EMB_DIR")")")_$(basename "$(dirname "$EMB_DIR")")_$(basename "$EMB_DIR")
M1="bayesian_framework/matrices/${TAG}_${SPLIT}_knn_dist_k${K}.npz"

echo "EMB_DIR=$EMB_DIR SPLIT=$SPLIT K=$K METHOD=$METHOD DELTA=$DELTA"
echo "matrix1 -> $M1"

# step 1: distances (skip if already built for this K/split)
if [ -f "$M1" ]; then
  echo "matrix 1 exists, skipping"
else
  python -u bayesian_framework/distance.py --embeddings_dir "$EMB_DIR" --split "$SPLIT" --k "$K"
fi

# step 2: figures as .npy
python -u bayesian_framework/similarity.py --matrix1 "$M1" --method "$METHOD" --delta "$DELTA" --npy

# step 3: figure
SIM="bayesian_framework/matrices/${TAG}_${SPLIT}_knn_sim_${METHOD}_k${K}_sim.npy"
python -u bayesian_framework/visualize.py --sim "$SIM" --labels "$EMB_DIR/${SPLIT}_labels.npy"

echo "done."
