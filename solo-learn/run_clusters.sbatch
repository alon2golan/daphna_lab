#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=32g
#SBATCH -c 8
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm_logs/clusters_%j.out
#SBATCH --error=slurm_logs/clusters_%j.err
#SBATCH --killable
#SBATCH --requeue

# Per-class sub-cluster count across all 4 pretrained runs (simclr/byol x
# cifar10/cifar100), at first/middle/last checkpoints. Extracts frozen backbone
# features and runs HDBSCAN on each class on its own -> how many density clusters
# a class fractures into, plus noise fraction.
#
# Usage:  sbatch run_clusters.sbatch
# Output: solo-learn/analysis/  (clusters.csv, clusters_dist_*.png,
#         clusters_per_class_*.png)

set -e

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
DATA_DIR="$ROOT/data"
cd "$ROOT"
source "$ROOT/venv/bin/activate"
cd "$ROOT/solo-learn"
mkdir -p slurm_logs

python -u analyze_clusters.py \
  --runs \
    trained_models/simclr/cifar10/1000ep \
    trained_models/simclr/cifar100/1000ep \
    trained_models/byol/cifar10/1000ep \
    trained_models/byol/cifar100/1000ep \
  --data_path "$DATA_DIR" \
  --epochs 0 499 999 \
  --output_dir analysis
