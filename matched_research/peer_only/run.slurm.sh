#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --job-name=npc-peer-only
set -euo pipefail

phase=$1
root=$2
reference=$3
project=$(dirname "$reference")
source="$reference/source"
upstream="$reference/active_learning_with_label_noise-733651bb0943a96392895103eb2ffc4e99a7c7d1"
python="$project/daphna_lab/.venv-vast/bin/python"
export PYTHONPATH="$reference/runtime_deps:$source:$project/vast_followup_20261024/runtime_deps:$project/vast_expanded_20261019/runtime_deps"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="$root/matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"

conditions=(clean sym30 sym40 sym50 asym30 asym40 asym50)
seeds=(71 72 73)
task=${SLURM_ARRAY_TASK_ID:-3}
case_name="${conditions[$((task % 7))]}_s${seeds[$((task / 7))]}"
case "$phase" in
    canary)
        test "$case_name" = sym50_s71
        budgets=(00400)
        ;;
    grid)
        test -s "$root/peer_only/sym50_s71/00400/result.json"
        budgets=(00400 00800 01200 01600 02000)
        ;;
    *) exit 2 ;;
esac

archive="$reference/runs/$case_name"
"$python" - "$archive/provenance.json" <<'PY'
import importlib.metadata
import json
import sys
from pathlib import Path

import torch

from run_official import file_sha256
import run_official

saved = json.loads(Path(sys.argv[1]).read_text())
gpu = torch.cuda.get_device_name()
if saved['gpu'] != 'NVIDIA GeForce RTX 2080 Ti' or gpu != saved['gpu']:
    raise RuntimeError(f'Expected archived GPU {saved["gpu"]}, got {gpu}')
packages = {name: importlib.metadata.version(name) for name in saved['packages']}
if packages != saved['packages'] or file_sha256(Path(run_official.__file__)) != saved['wrapper_sha256']:
    raise RuntimeError('The runtime packages or pinned fit wrapper differ from the reference')
print(json.dumps(dict(gpu=gpu, packages=packages, wrapper_sha256=saved['wrapper_sha256'])), flush=True)
PY

for budget in "${budgets[@]}"; do
    if [[ "$phase:$case_name:$budget" == grid:sym50_s71:00400 ]]; then
        continue
    fi
    "$python" -u "$source/run_official.py" fit --upstream "$upstream" --inputs "$reference/inputs" \
        --snapshot "$archive/fits/$budget" --output "$root/peer_only/$case_name/$budget" \
        --mask-function "$root/peer_only.py:peer_only_mask"
done
