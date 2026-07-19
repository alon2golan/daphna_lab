#!/bin/bash
#SBATCH -p medium
#SBATCH --mem=48g
#SBATCH -c 4
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=hierarchical_framework/outputs/slurm_logs/tin_%j.out
#SBATCH --error=hierarchical_framework/outputs/slurm_logs/tin_%j.err
#SBATCH --killable
#SBATCH --requeue

# ARCHIVED: TinyImageNet path on the old (itai's SimCLR) embeddings. The
# TinyImageNet results were incorrect; kept under archive/ for reference only.
#
# Full TinyImageNet ProbCover pipeline:
#   1) build the embeddings dir (good features_seed1 + held-out eval split)
#   2) build the WordNet superclass map (coarse_map.npy)
#   3) pick delta with deep_al's knee algorithm (once), reuse for both runs
#   4) run_active_learning -> fine-accuracy ProbCover curve
#   5) accuracy_vs_budget   -> fine + coarse + oracle + near/far, with the coarse map
# Everything logs to the wandb project applying_priors.
#
# Usage:  sbatch hierarchical_framework/archive/scripts/run_tinyimagenet.sbatch [BUDGET] [ROUNDS]

set -e
BUDGET=${1:-200}         # points labeled per round (200 = 1 label/class)
ROUNDS=${2:-10}

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
cd "$ROOT"
source "$ROOT/venv/bin/activate"          # full stack + wandb
mkdir -p hierarchical_framework/outputs/slurm_logs

EMB=hierarchical_framework/embeddings/old_embeddings/tinyimagenet_simclr
CM=$EMB/coarse_map.npy

echo "== step 1: build embeddings dir =="
python -u hierarchical_framework/archive/data_prep/prepare_tinyimagenet.py
echo "== step 2: build WordNet coarse map =="
python -u hierarchical_framework/archive/data_prep/build_tinyimagenet_coarse.py

echo "== step 3: knee delta (deep_al) =="
# compute once so run_active_learning and accuracy_vs_budget use the identical value
DELTA=$(python -c "import sys, numpy as np; sys.path.insert(0, 'hierarchical_framework/pipeline'); \
from delta_selection import resolve_delta; \
f=np.load('$EMB/train_features_normalized.npy'); \
print('DELTA=%.6f' % resolve_delta('knee', f, 200))" | grep '^DELTA=' | cut -d= -f2)
echo "resolved knee delta = $DELTA"

echo "== step 4: ProbCover fine-accuracy curve =="
python -u hierarchical_framework/pipeline/run_active_learning.py --embeddings_dir "$EMB" --strategy probcover \
  --delta "$DELTA" --budget "$BUDGET" --rounds "$ROUNDS" \
  --wandb --project applying_priors --name tin_probcover_knee_b${BUDGET}

echo "== step 5: fine + coarse + oracle analysis =="
python -u hierarchical_framework/analysis/accuracy_vs_budget.py --embeddings_dir "$EMB" --coarse_map "$CM" \
  --delta "$DELTA" --budget "$BUDGET" --rounds "$ROUNDS" \
  --wandb --project applying_priors --name tin_erroranalysis_knee_b${BUDGET}

echo "done. delta=$DELTA"
