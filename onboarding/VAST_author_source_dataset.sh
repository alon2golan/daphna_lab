#!/bin/bash
# Explicit experiment using the copied author propagation helper + lab stop rule.
#SBATCH --mem=40g
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --gres=gg:g0:1,vmem:8g
#SBATCH --exclude=arion-01,binky-01,creek-01,drape-01,dumfries-004,firefoot-01,gringolet-01,incitatus-01,khan-01
#SBATCH --output=vast_author_source_%j.log
#SBATCH --error=vast_author_source_%j.log
set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-/cs/labs/daphna/$USER/projects/daphna_lab}"
VAST_PYTHON="${VAST_PYTHON:-$REPO_DIR/.venv-vast/bin/python}"
VAST_DEVICE="${VAST_DEVICE:-cuda}"
SEED="${SEED:-0}"
SENDER_THRESHOLD="${SENDER_THRESHOLD:-0.05}"
RECEIVER_THRESHOLD="${RECEIVER_THRESHOLD:-0.75}"
MAX_ROUNDS="${MAX_ROUNDS:-50}"
: "${VAST_MANIFEST:?Set VAST_MANIFEST to a verified dataset manifest}"
export VAST_FAISS_THREADS="${VAST_FAISS_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
cd "$REPO_DIR"
"$VAST_PYTHON" -u deep_al/tools/train_vast.py --manifest "$VAST_MANIFEST" \
  --labels-per-class 4 --seed "$SEED" --epochs 300 --device "$VAST_DEVICE" \
  --propagation author-source --sender-threshold "$SENDER_THRESHOLD" \
  --receiver-threshold "$RECEIVER_THRESHOLD" --max-rounds "$MAX_ROUNDS" --preflight
dataset="$("$VAST_PYTHON" -c 'import sys; from deep_al.vast.data import read_config; print(read_config(sys.argv[1])["dataset"])' "$VAST_MANIFEST")"
VAST_OUTPUT_DIR="${VAST_OUTPUT_DIR:-$REPO_DIR/results/vast/${dataset}_author_source_seed${SEED}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
VAST_CACHE_DIR="${VAST_CACHE_DIR:-$VAST_OUTPUT_DIR/graph_cache}"
if [[ -e "$VAST_OUTPUT_DIR" ]]; then
  echo "Output already exists: $VAST_OUTPUT_DIR. Choose a new VAST_OUTPUT_DIR." >&2
  exit 2
fi
mkdir -p "$VAST_OUTPUT_DIR" "$VAST_CACHE_DIR"
echo "VAST author-source | host=$(hostname) | dataset=$dataset | seed=$SEED | device=$VAST_DEVICE | sender_threshold=$SENDER_THRESHOLD | receiver_threshold=$RECEIVER_THRESHOLD | max_rounds=$MAX_ROUNDS | cache=$VAST_CACHE_DIR | output=$VAST_OUTPUT_DIR"
for budget in 1 2 4; do
  "$VAST_PYTHON" -u deep_al/tools/train_vast.py --manifest "$VAST_MANIFEST" \
    --labels-per-class "$budget" --seed "$SEED" --epochs 300 --device "$VAST_DEVICE" \
    --propagation author-source --sender-threshold "$SENDER_THRESHOLD" \
    --receiver-threshold "$RECEIVER_THRESHOLD" --max-rounds "$MAX_ROUNDS" \
    --cache-dir "$VAST_CACHE_DIR" \
    --output-dir "$VAST_OUTPUT_DIR/${dataset}_labels${budget}_seed${SEED}"
  "$VAST_PYTHON" - "$VAST_OUTPUT_DIR" <<'PY'
import csv
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
rows = [json.loads(path.read_text()) for path in sorted(root.glob("*_labels*_seed*/result.json"))]
with (root / "summary.csv.tmp").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(root / "summary.csv.tmp").replace(root / "summary.csv")
PY
done
echo "All three author-source budgets completed. Metrics: $VAST_OUTPUT_DIR/summary.csv"
