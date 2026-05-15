#!/bin/bash
#SBATCH --job-name=mh_density_sel
#SBATCH --mem=96g
#SBATCH -c 4
#SBATCH --time=1-00:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --output=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/diagnostics/density_selection/mh_density_selection-%j.txt
#SBATCH --error=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/diagnostics/density_selection/mh_density_selection-%j.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/maxherding/diagnostics/density_selection"
RESULTS_DIR="$PROJECT_DIR/results_summaries/maxherding_density_selection"

mkdir -p "$OUT_DIR" "$RESULTS_DIR"
cd "$PROJECT_DIR"

/cs/labs/daphna/ido.dotan/virtual_envs/my_env_3.11/bin/python \
  scripts/analyze_maxherding_density_selection.py \
  --datasets CIFAR10 CIFAR100 \
  --seeds 3 4 5 \
  --anchor-sample-size 5000 \
  --distance-batch-size 128 \
  --outdir "$RESULTS_DIR"
