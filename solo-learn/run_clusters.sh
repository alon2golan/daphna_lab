#!/usr/bin/env bash
# Portable (no-SLURM) version of run_clusters.sbatch -- runs on the cluster OR any
# regular computer. Per-class sub-cluster count for all 4 pretrained runs at
# first/middle/last checkpoints: extracts frozen features, runs HDBSCAN per class.
#
# Usage:  ./run_clusters.sh
# Output: solo-learn/analysis/  (clusters.csv, clusters_dist_*.png,
#         clusters_per_class_*.png)
#
# Uses the GPU automatically if torch sees one, otherwise runs on CPU (slower).

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # .../solo-learn
ROOT=$(dirname "$SCRIPT_DIR")                              # project root
cd "$SCRIPT_DIR"

[ -f "$ROOT/venv/bin/activate" ] && source "$ROOT/venv/bin/activate"

DATA_DIR="${DATA_DIR:-$ROOT/data}"
[ -d "$DATA_DIR" ] || DATA_DIR="$SCRIPT_DIR/datasets"

echo "Counting per-class sub-clusters (data=$DATA_DIR)"

python -u analyze_clusters.py \
  --runs \
    trained_models/simclr/cifar10/1000ep \
    trained_models/simclr/cifar100/1000ep \
    trained_models/byol/cifar10/1000ep \
    trained_models/byol/cifar100/1000ep \
  --data_path "$DATA_DIR" \
  --epochs 0 499 999 \
  --output_dir analysis
