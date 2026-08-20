#!/bin/bash
#SBATCH --mem=1g
#SBATCH -c 1
#SBATCH --time=00:10:00
#SBATCH --mail-user=tom.novak@mail.huji.ac.il
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=gpu_all_noisy_al_experiments_%j.log
#SBATCH --error=gpu_all_noisy_al_experiments_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1

# One entry point for every unique scientific run in the current project.
# Smoke tests, missing-run repair scripts, rerun-only scripts, and the earlier
# partial Netta script are excluded because they are subsets, duplicates, or
# superseded by the suites below.
#
# Default total:
#   448 regular main-grid runs
#   112 additional asymmetric-noise=0.3 runs
#    50 Netta-aligned cold-start/Framework-2 AUM runs
#   610 unique planned runs
#
# By default this lightweight launcher splits each suite by final-label budget
# and submits six independent GPU jobs for more parallelism:
#   regular 400 (224), regular 800 (224), asymmetric 0.3 400 (56),
#   asymmetric 0.3 800 (56), Netta 400 (25), and Netta 800 (25).
# Use LAUNCH_MODE=sequential to execute the same six shards in this process.

REPO_DIR="/cs/labs/daphna/$USER/projects/daphna_lab"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export RUN_TIMESTAMP

RUN_REGULAR_GRID="${RUN_REGULAR_GRID:-1}"
RUN_ASYM_03_GRID="${RUN_ASYM_03_GRID:-1}"
RUN_NETTA_ALIGNED_GRID="${RUN_NETTA_ALIGNED_GRID:-1}"
LAUNCH_MODE="${LAUNCH_MODE:-submit}"

if [[ "$LAUNCH_MODE" != "submit" && "$LAUNCH_MODE" != "sequential" ]]; then
  echo "LAUNCH_MODE must be submit or sequential; got: $LAUNCH_MODE" >&2
  exit 2
fi

TOTAL_RUNS=0
TOTAL_LAUNCHES=0
if [[ "$RUN_REGULAR_GRID" == "1" ]]; then
  TOTAL_RUNS=$((TOTAL_RUNS + 448))
  TOTAL_LAUNCHES=$((TOTAL_LAUNCHES + 2))
fi
if [[ "$RUN_ASYM_03_GRID" == "1" ]]; then
  TOTAL_RUNS=$((TOTAL_RUNS + 112))
  TOTAL_LAUNCHES=$((TOTAL_LAUNCHES + 2))
fi
if [[ "$RUN_NETTA_ALIGNED_GRID" == "1" ]]; then
  TOTAL_RUNS=$((TOTAL_RUNS + 50))
  TOTAL_LAUNCHES=$((TOTAL_LAUNCHES + 2))
fi

echo "Starting all project experiments"
echo "Planned runs: $TOTAL_RUNS"
echo "Planned launches: $TOTAL_LAUNCHES"
echo "Regular 448 grid enabled: $RUN_REGULAR_GRID"
echo "Asymmetric 0.3 grid enabled: $RUN_ASYM_03_GRID"
echo "Netta-aligned 50-run grid enabled: $RUN_NETTA_ALIGNED_GRID"
echo "Launch mode: $LAUNCH_MODE"
echo "Timestamp: $RUN_TIMESTAMP"

run_suite () {
  local LABEL="$1"
  local SCRIPT_PATH="$2"
  local RUN_400="$3"
  local RUN_800="$4"
  if [[ "$LAUNCH_MODE" == "submit" ]]; then
    echo "Submitting $LABEL"
    sbatch \
      --export=ALL,RUN_TIMESTAMP="$RUN_TIMESTAMP",RUN_FINAL_400="$RUN_400",RUN_FINAL_800="$RUN_800" \
      "$SCRIPT_PATH"
  else
    echo "Running $LABEL sequentially"
    env \
      RUN_TIMESTAMP="$RUN_TIMESTAMP" \
      RUN_FINAL_400="$RUN_400" \
      RUN_FINAL_800="$RUN_800" \
      bash "$SCRIPT_PATH"
  fi
}

if [[ "$RUN_REGULAR_GRID" == "1" ]]; then
  echo
  run_suite "regular grid, final size 400 (224 runs)" \
    "$REPO_DIR/onboarding/NAL_grid_448_runs.sh" 1 0
  run_suite "regular grid, final size 800 (224 runs)" \
    "$REPO_DIR/onboarding/NAL_grid_448_runs.sh" 0 1
fi

if [[ "$RUN_ASYM_03_GRID" == "1" ]]; then
  echo
  run_suite "asymmetric-noise=0.3 grid, final size 400 (56 runs)" \
    "$REPO_DIR/onboarding/NAL_grid_script_all_asym_03_400_800_seed42.sh" 1 0
  run_suite "asymmetric-noise=0.3 grid, final size 800 (56 runs)" \
    "$REPO_DIR/onboarding/NAL_grid_script_all_asym_03_400_800_seed42.sh" 0 1
fi

if [[ "$RUN_NETTA_ALIGNED_GRID" == "1" ]]; then
  echo
  run_suite "Netta-aligned grid, final size 400 (25 runs)" \
    "$REPO_DIR/onboarding/NAL_grid_netta_aligned_aum_400_800.sh" 1 0
  run_suite "Netta-aligned grid, final size 800 (25 runs)" \
    "$REPO_DIR/onboarding/NAL_grid_netta_aligned_aum_400_800.sh" 0 1
fi

echo
if [[ "$LAUNCH_MODE" == "submit" ]]; then
  echo "All enabled experiment suites were submitted"
  echo "Submitted planned configurations: $TOTAL_RUNS"
  echo "Submitted jobs: $TOTAL_LAUNCHES"
else
  echo "All enabled experiment suites finished"
  echo "Executed planned configurations: $TOTAL_RUNS"
  echo "Executed shards: $TOTAL_LAUNCHES"
fi
echo "Finished at $(date)"
