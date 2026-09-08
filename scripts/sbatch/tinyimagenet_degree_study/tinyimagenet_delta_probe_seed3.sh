#!/bin/bash
#SBATCH --job-name=tiny_delta_s3
#SBATCH --mem=96g
#SBATCH -c 4
#SBATCH --time=1-00:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --output=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/tinyimagenet_degree_study/delta/tinyimagenet_delta_seed3-%j.txt
#SBATCH --error=/cs/labs/daphna/ido.dotan/projects/daphna_lab/outputs/maxherding/tinyimagenet_degree_study/delta/tinyimagenet_delta_seed3-%j.err
#SBATCH --killable

set -euo pipefail

PROJECT_DIR=/cs/labs/daphna/ido.dotan/projects/daphna_lab
OUT_DIR="$PROJECT_DIR/outputs/maxherding/tinyimagenet_degree_study/delta"
RESULTS_DIR="$PROJECT_DIR/results_summaries/maxherding_tinyimagenet_density_selection"
SEED=3
MAX_POINTS=${MAX_POINTS_OVERRIDE:-100000}

mkdir -p "$OUT_DIR" "$RESULTS_DIR"
cd "$PROJECT_DIR"

/cs/labs/daphna/ido.dotan/virtual_envs/my_env_3.11/bin/python -c "
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '$PROJECT_DIR/deep_al')
from pycls.datasets.utils.features import load_features
from pycls.datasets.utils.select_delta import select_delta

seed = $SEED
max_points = int('$MAX_POINTS')
features = load_features('TINYIMAGENET', train=True, normalized=True).astype(np.float32, copy=False)
if max_points > 0 and max_points < len(features):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(features), size=max_points, replace=False))
    features_for_delta = features[idx]
else:
    features_for_delta = features

delta = select_delta(features_for_delta, num_classes=200, seed=seed, debug=True)
out = Path('$RESULTS_DIR') / 'tinyimagenet_delta_seed3.txt'
out.write_text(f'delta={delta:.6f}\\nseed={seed}\\nnum_points={len(features_for_delta)}\\n', encoding='ascii')
print(f'TINYIMAGENET_DELTA={delta:.6f}')
print(f'Wrote {out}')
"
