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

# One entry point for all 610 unique scientific runs. The launcher creates four
# balanced workers. Each worker receives one regular-grid noise type (112
# runs), 28 asymmetric-0.3 runs, and 12 or 13 Netta-aligned runs:
#
#   shard 0: clean regular grid      112 + 28 + 13 = 153
#   shard 1: symmetric regular grid  112 + 28 + 13 = 153
#   shard 2: asymmetric regular grid 112 + 28 + 12 = 152
#   shard 3: CIFAR-100N regular grid 112 + 28 + 12 = 152
#
# Smoke tests and repair/rerun-only scripts are excluded because they duplicate
# configurations in these three complete suites.

REPO_DIR="/cs/labs/daphna/$USER/projects/daphna_lab"
MASTER_SCRIPT="$REPO_DIR/onboarding/NAL_run_all_project_experiments.sh"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_REGULAR_GRID="${RUN_REGULAR_GRID:-1}"
RUN_ASYM_03_GRID="${RUN_ASYM_03_GRID:-1}"
RUN_NETTA_ALIGNED_GRID="${RUN_NETTA_ALIGNED_GRID:-1}"
LAUNCH_MODE="${LAUNCH_MODE:-submit}"
PROJECT_SHARD_INDEX="${PROJECT_SHARD_INDEX:-}"

for FLAG_NAME in RUN_REGULAR_GRID RUN_ASYM_03_GRID RUN_NETTA_ALIGNED_GRID; do
  FLAG_VALUE="${!FLAG_NAME}"
  if [[ "$FLAG_VALUE" != "0" && "$FLAG_VALUE" != "1" ]]; then
    echo "$FLAG_NAME must be 0 or 1; got: $FLAG_VALUE" >&2
    exit 2
  fi
done
if [[ "$LAUNCH_MODE" != "submit" && "$LAUNCH_MODE" != "sequential" ]]; then
  echo "LAUNCH_MODE must be submit or sequential; got: $LAUNCH_MODE" >&2
  exit 2
fi
if [[ -n "$PROJECT_SHARD_INDEX" && ! "$PROJECT_SHARD_INDEX" =~ ^[0-3]$ ]]; then
  echo "PROJECT_SHARD_INDEX must be empty or an integer from 0 to 3" >&2
  exit 2
fi

SHARD_NOISE_TYPES=(clean sym asym cifar100n)
SHARD_NETTA_RUNS=(13 13 12 12)

worker_run_count () {
  local SHARD_INDEX="$1"
  local COUNT=0
  if [[ "$RUN_REGULAR_GRID" == "1" ]]; then
    COUNT=$((COUNT + 112))
  fi
  if [[ "$RUN_ASYM_03_GRID" == "1" ]]; then
    COUNT=$((COUNT + 28))
  fi
  if [[ "$RUN_NETTA_ALIGNED_GRID" == "1" ]]; then
    COUNT=$((COUNT + SHARD_NETTA_RUNS[SHARD_INDEX]))
  fi
  echo "$COUNT"
}

run_worker () {
  local SHARD_INDEX="$1"
  local SHARD_NOISE="${SHARD_NOISE_TYPES[$SHARD_INDEX]}"
  local SHARD_RUNS
  SHARD_RUNS="$(worker_run_count "$SHARD_INDEX")"

  echo "Starting balanced worker $SHARD_INDEX/3"
  echo "Planned runs: $SHARD_RUNS"
  echo "Regular noise type: $SHARD_NOISE"
  echo "Timestamp: $RUN_TIMESTAMP"

  if [[ "$RUN_REGULAR_GRID" == "1" ]]; then
    env \
      RUN_TIMESTAMP="$RUN_TIMESTAMP" \
      PROJECT_SHARD_INDEX="$SHARD_INDEX" \
      GRID_NOISE_TYPES="$SHARD_NOISE" \
      RUN_FINAL_400=1 \
      RUN_FINAL_800=1 \
      bash "$REPO_DIR/onboarding/NAL_grid_448_runs.sh"
  fi

  if [[ "$RUN_ASYM_03_GRID" == "1" ]]; then
    env \
      RUN_TIMESTAMP="$RUN_TIMESTAMP" \
      PROJECT_SHARD_INDEX="$SHARD_INDEX" \
      RUN_FINAL_400=1 \
      RUN_FINAL_800=1 \
      bash "$REPO_DIR/onboarding/NAL_grid_script_all_asym_03_400_800_seed42.sh"
  fi

  if [[ "$RUN_NETTA_ALIGNED_GRID" == "1" ]]; then
    env \
      RUN_TIMESTAMP="$RUN_TIMESTAMP" \
      PROJECT_SHARD_INDEX="$SHARD_INDEX" \
      RUN_FINAL_400=1 \
      RUN_FINAL_800=1 \
      bash "$REPO_DIR/onboarding/NAL_grid_netta_aligned_aum_400_800.sh"
  fi

  echo "Balanced worker $SHARD_INDEX finished $SHARD_RUNS planned runs at $(date)"
}

# A submitted worker re-enters this script with its shard index and executes
# its three assigned portions sequentially on one GPU.
if [[ -n "$PROJECT_SHARD_INDEX" ]]; then
  run_worker "$PROJECT_SHARD_INDEX"
  exit 0
fi

TOTAL_RUNS=0
if [[ "$RUN_REGULAR_GRID" == "1" ]]; then
  TOTAL_RUNS=$((TOTAL_RUNS + 448))
fi
if [[ "$RUN_ASYM_03_GRID" == "1" ]]; then
  TOTAL_RUNS=$((TOTAL_RUNS + 112))
fi
if [[ "$RUN_NETTA_ALIGNED_GRID" == "1" ]]; then
  TOTAL_RUNS=$((TOTAL_RUNS + 50))
fi
if [[ "$TOTAL_RUNS" -eq 0 ]]; then
  echo "No experiment suites are enabled" >&2
  exit 2
fi

echo "Starting all project experiments"
echo "Planned runs: $TOTAL_RUNS"
echo "Balanced launches: 4"
echo "Launch mode: $LAUNCH_MODE"
echo "Timestamp: $RUN_TIMESTAMP"

for SHARD_INDEX in 0 1 2 3; do
  SHARD_RUNS="$(worker_run_count "$SHARD_INDEX")"
  SHARD_NOISE="${SHARD_NOISE_TYPES[$SHARD_INDEX]}"
  if [[ "$LAUNCH_MODE" == "submit" ]]; then
    echo "Submitting shard $SHARD_INDEX: $SHARD_RUNS runs, regular noise=$SHARD_NOISE"
    sbatch \
      --job-name="noisy_al_${SHARD_INDEX}" \
      --mem=40g \
      --cpus-per-task=8 \
      --time=96:00:00 \
      --gres=gg:g0:1,vmem:40g \
      --output="gpu_all_noisy_al_shard_${SHARD_INDEX}_%j.log" \
      --error="gpu_all_noisy_al_shard_${SHARD_INDEX}_%j.log" \
      --export=ALL,RUN_TIMESTAMP="$RUN_TIMESTAMP",PROJECT_SHARD_INDEX="$SHARD_INDEX",RUN_REGULAR_GRID="$RUN_REGULAR_GRID",RUN_ASYM_03_GRID="$RUN_ASYM_03_GRID",RUN_NETTA_ALIGNED_GRID="$RUN_NETTA_ALIGNED_GRID" \
      "$MASTER_SCRIPT"
  else
    echo "Running shard $SHARD_INDEX sequentially: $SHARD_RUNS runs"
    env \
      RUN_TIMESTAMP="$RUN_TIMESTAMP" \
      PROJECT_SHARD_INDEX="$SHARD_INDEX" \
      RUN_REGULAR_GRID="$RUN_REGULAR_GRID" \
      RUN_ASYM_03_GRID="$RUN_ASYM_03_GRID" \
      RUN_NETTA_ALIGNED_GRID="$RUN_NETTA_ALIGNED_GRID" \
      bash "$MASTER_SCRIPT"
  fi
done

if [[ "$LAUNCH_MODE" == "submit" ]]; then
  echo "Submitted four balanced jobs containing $TOTAL_RUNS configurations"
else
  echo "Finished four sequential shards containing $TOTAL_RUNS configurations"
fi
echo "Finished at $(date)"
