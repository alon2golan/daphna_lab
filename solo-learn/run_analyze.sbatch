#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=32g
#SBATCH -c 8
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm_logs/analyze_%j.out
#SBATCH --error=slurm_logs/analyze_%j.err
#SBATCH --killable
#SBATCH --requeue

# Analyze the evolution of every saved checkpoint across all 4 pretrained runs
# (simclr/byol x cifar10/cifar100). Extracts frozen backbone features, computes
# representation-quality metrics, and plots metric-vs-epoch curves + UMAP snapshots.
#
# Usage:  sbatch run_analyze.sbatch
# Output: solo-learn/analysis/  (metrics.csv, metric_*_vs_epoch.png, umap_*.png)

set -e

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
DATA_DIR="$ROOT/data"
cd "$ROOT"
source "$ROOT/venv/bin/activate"
cd "$ROOT/solo-learn"
mkdir -p slurm_logs

python -u analyze_checkpoints.py \
  --runs \
    trained_models/simclr/cifar10/1000ep \
    trained_models/simclr/cifar100/1000ep \
    trained_models/byol/cifar10/1000ep \
    trained_models/byol/cifar100/1000ep \
  --data_path "$DATA_DIR" \
  --umap_epochs 0 499 999 \
  --output_dir analysis
