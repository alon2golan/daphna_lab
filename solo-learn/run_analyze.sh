#!/usr/bin/env bash
# Portable (no-SLURM) version of run_analyze.sbatch -- runs on the cluster OR any
# regular computer. Analyzes every saved checkpoint of all 4 pretrained runs:
# extracts frozen features, computes representation metrics, plots evolution + UMAP.
#
# Usage:  ./run_analyze.sh
# Output: solo-learn/analysis/  (metrics.csv, metric_*_vs_epoch.png, umap_*.png)
#
# Uses the GPU automatically if torch sees one, otherwise runs on CPU (slower).

set -e

# Resolve paths from THIS script's location, so it works from anywhere / any machine.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # .../solo-learn
ROOT=$(dirname "$SCRIPT_DIR")                              # project root
cd "$SCRIPT_DIR"

# Activate the project venv only if it exists; otherwise use whatever python is on PATH.
[ -f "$ROOT/venv/bin/activate" ] && source "$ROOT/venv/bin/activate"

# Prefer the lab's pre-downloaded CIFAR under <root>/data; else use ./datasets
# (torchvision downloads there automatically on first run).
DATA_DIR="${DATA_DIR:-$ROOT/data}"
[ -d "$DATA_DIR" ] || DATA_DIR="$SCRIPT_DIR/datasets"

echo "Analyzing checkpoints (data=$DATA_DIR)"

python -u analyze_checkpoints.py \
  --runs \
    trained_models/simclr/cifar10/1000ep \
    trained_models/simclr/cifar100/1000ep \
    trained_models/byol/cifar10/1000ep \
    trained_models/byol/cifar100/1000ep \
  --data_path "$DATA_DIR" \
  --umap_epochs 0 499 999 \
  --output_dir analysis
