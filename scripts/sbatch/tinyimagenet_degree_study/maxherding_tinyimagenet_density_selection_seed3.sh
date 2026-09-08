#!/bin/bash
#SBATCH --job-name=mh_tiny_diag
#SBATCH --mem=128g
#SBATCH -c 4
#SBATCH --time=1-12:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --output=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/tinyimagenet_degree_study/diagnostics/maxherding_tinyimagenet_density_selection-%j.txt
#SBATCH --error=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/tinyimagenet_degree_study/diagnostics/maxherding_tinyimagenet_density_selection-%j.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/maxherding/tinyimagenet_degree_study/diagnostics"
RESULTS_DIR="$PROJECT_DIR/results_summaries/maxherding_tinyimagenet_density_selection"

mkdir -p "$OUT_DIR" "$RESULTS_DIR"
cd "$PROJECT_DIR"

/cs/labs/daphna/ido.dotan/virtual_envs/my_env_3.11/bin/python \
  scripts/analyze_maxherding_density_selection.py \
  --datasets TINYIMAGENET \
  --seeds 3 \
  --run-glob "output/TINYIMAGENET/resnet18/*/maxherding_tinyimagenet_degree_*" \
  --name-regex "maxherding_tinyimagenet_degree_.*_(global_sigma|local_sigma_k[0-9]+)" \
  --anchor-sample-size 10000 \
  --distance-batch-size 64 \
  --sigma-batch-size 256 \
  --outdir "$RESULTS_DIR"
