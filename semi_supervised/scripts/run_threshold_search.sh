#!/usr/bin/env bash
# Rank propagation thresholds for every case, WITHOUT training. For each
# {simclr,byol} x {cifar10,cifar100} x per-class budget L, it sweeps
# (kernel, bandwidth, sim_threshold, conf_threshold) on the frozen features and
# prints a ranked table + a recommended shortlist, and saves the full grid to
#   results/<dataset>/<embedding>/threshold_search/L<L>.json
# so you can decide which thresholds are worth an actual SupCon run.
#
# Uses the cached base_features from the baseline jobs when present; otherwise
# extracts fresh from the checkpoint. Cheap enough to run on one node.
#
# Usage (from solo-learn):
#   ./run_threshold_search.sh                 # budgets 2,4,6,10,25,50
#   ./run_threshold_search.sh 10,25,50        # only larger budgets
# Env: SEED (default 1); GRID overrides the swept grid flags, e.g.
#   GRID="--bandwidths auto,0.5x,0.25x --sim_thresholds 0.0,0.1,0.2 --conf_thresholds 0.6,0.8,0.9"

set -e

BUDGETS=${1:-${BUDGETS:-2,4,6,10,25,50}}
SEED=${SEED:-1}
GRID=${GRID:-"--bandwidths auto,0.5x,0.25x,0.1x --sim_thresholds 0.0,0.1,0.2,0.3 --conf_thresholds 0.8,0.9,0.95 --min_purity 0.9 --wrong_penalty 3.0"}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # .../semi_supervised/scripts
ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")
cd "$ROOT"
[ -f "$ROOT/venv/bin/activate" ] && source "$ROOT/venv/bin/activate"
RESULTS_ROOT="$ROOT/semi_supervised/results"
DATA_DIR="${DATA_DIR:-$ROOT/data}"

for METHOD in simclr byol; do
  for DATASET in cifar10 cifar100; do
    case "$DATASET" in cifar10) NCLASSES=10 ;; cifar100) NCLASSES=100 ;; esac
    CASE="$RESULTS_ROOT/$DATASET/$METHOD"
    FEATS="$CASE/base_features"
    if [ -f "$FEATS/train_features.npy" ]; then
      SRC="--features $FEATS"
    else
      SRC="--base_checkpoint $ROOT/solo-learn/trained_models/$METHOD/$DATASET/1000ep"
    fi
    IFS=',' read -ra LS <<< "$BUDGETS"
    for L in "${LS[@]}"; do
      OUT="$CASE/threshold_search/L${L}.json"
      echo "================ $METHOD $DATASET L=$L ================"
      python -u -m semi_supervised.old.threshold_search \
        $SRC --dataset "$DATASET" --data_dir "$DATA_DIR" \
        --per_class_budget "$L" --seed "$SEED" \
        --kernels gaussian,laplacian,cosine $GRID --out "$OUT" \
        2>/dev/null
    done
  done
done

echo "reports under: $RESULTS_ROOT/<dataset>/<embedding>/threshold_search/L<L>.json"
