#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --gres=gg:g0:1,vmem:8g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --job-name=signed-aum-audit
set -euo pipefail

root=$1
geometry=$2
upstream=$3
projects=/cs/labs/daphna/tomn/projects
python=$projects/daphna_lab/.venv-vast/bin/python
npc=$projects/cifar_matched_npc_20261127

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$npc/runtime_deps:$projects/vast_followup_20261024/runtime_deps:$projects/vast_expanded_20261019/runtime_deps${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=2 MKL_NUM_THREADS=4 MPLCONFIGDIR="$root/matplotlib_cache"
cd "$root"
"$python" -u source/audit.py run "$SLURM_ARRAY_TASK_ID" "$geometry" "$upstream" "seed_${SLURM_ARRAY_TASK_ID}.json"
