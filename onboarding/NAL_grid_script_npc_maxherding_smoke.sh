#!/bin/bash
#SBATCH --mem=40g
#SBATCH -c 8
#SBATCH --time=08:00:00
#SBATCH --gres=gg:g0:1,vmem:40g
#SBATCH --mail-user=tom.novak@mail.huji.ac.il
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=gpu_grid_npc_maxherding_smoke_%j.log
#SBATCH --error=gpu_grid_npc_maxherding_smoke_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GRID_NOISE_TYPES="${GRID_NOISE_TYPES:-sym,asym}"
export GRID_AL_METHODS="${GRID_AL_METHODS:-maxherding,npc_maxherding_oracle,npc_maxherding_crossval}"
export MAX_ITER="${MAX_ITER:-1}"
export DEBUG_JSON="${DEBUG_JSON:-1}"
export RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)_smoke}"

bash "$SCRIPT_DIR/NAL_grid_script_npc_maxherding_seed42.sh"
