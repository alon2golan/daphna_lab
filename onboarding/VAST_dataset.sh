#!/bin/bash
# One dataset, three full label budgets, one seed. Each dataset can run separately.
#SBATCH --mem=40g
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --gres=gg:g0:1,vmem:8g
#SBATCH --output=vast_dataset_%j.log
#SBATCH --error=vast_dataset_%j.log
set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-/cs/labs/daphna/$USER/projects/daphna_lab}"
VAST_PYTHON="${VAST_PYTHON:-$REPO_DIR/.venv-vast/bin/python}"
VAST_DEVICE="${VAST_DEVICE:-cuda}"
SEED="${SEED:-0}"
: "${VAST_MANIFEST:?Set VAST_MANIFEST to a verified dataset manifest}"
export VAST_FAISS_THREADS="${VAST_FAISS_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
cd "$REPO_DIR"
"$VAST_PYTHON" -u deep_al/tools/train_vast.py --manifest "$VAST_MANIFEST" \
  --labels-per-class 4 --seed "$SEED" --epochs 300 --device "$VAST_DEVICE" --preflight
dataset="$("$VAST_PYTHON" -c 'import sys; from deep_al.vast.data import read_config; print(read_config(sys.argv[1])["dataset"])' "$VAST_MANIFEST")"
VAST_OUTPUT_DIR="${VAST_OUTPUT_DIR:-$REPO_DIR/results/vast/${dataset}_seed${SEED}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
if [[ -e "$VAST_OUTPUT_DIR" ]]; then
  echo "Output already exists: $VAST_OUTPUT_DIR. Choose a new VAST_OUTPUT_DIR." >&2
  exit 2
fi
echo "VAST full dataset | host=$(hostname) | dataset=$dataset | seed=$SEED | output=$VAST_OUTPUT_DIR"
for budget in 1 2 4; do
  "$VAST_PYTHON" -u deep_al/tools/train_vast.py --manifest "$VAST_MANIFEST" \
    --labels-per-class "$budget" --seed "$SEED" --epochs 300 --device "$VAST_DEVICE" \
    --propagation literal-paper --cache-dir "$VAST_OUTPUT_DIR/graph_cache" \
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
echo "All three budgets completed. Metrics: $VAST_OUTPUT_DIR/summary.csv"
