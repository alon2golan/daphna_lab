#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --gres=gg:g0:1,vmem:8g
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --job-name=npc-byol-matched
set -euo pipefail

phase=$1
root=$2
upstream=$3
inputs=$4
code="$root/source"
projects=/cs/labs/daphna/tomn/projects
python="${NPC_PYTHON:-$projects/daphna_lab/.venv-vast/bin/python}"
export PYTHONPATH="$root/runtime_deps:${NPC_RUNTIME_DEPS:-$projects/vast_followup_20261024/runtime_deps:$projects/vast_expanded_20261019/runtime_deps}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

conditions=(clean sym30 sym40 sym50 asym30 asym40 asym50)
seeds=(71 72 73)
task=${SLURM_ARRAY_TASK_ID:-3}
condition=${conditions[$((task % 7))]}
seed=${seeds[$((task / 7))]}
case_name="${condition}_s${seed}"

run_baseline() {
    mkdir -p "$root/runs"
    "$python" -u "$code/run_official.py" run --upstream "$upstream" --inputs "$inputs" \
        --output "$root/runs/$case_name" --condition "$condition" --seed "$seed"
}

run_peer() {
    test -s "$root/runs/$case_name/summary.json"
    mkdir -p "$root/peer/$case_name"
    for budget in 00400 00800 01200 01600 02000; do
        "$python" -u "$code/run_official.py" fit --upstream "$upstream" --inputs "$inputs" \
            --snapshot "$root/runs/$case_name/fits/$budget" \
            --output "$root/peer/$case_name/$budget" \
            --mask-function "$code/filter_swap.py:peer_aum_mask"
    done
}

case "$phase" in
    smoke)
        mkdir -p "$root/smoke"
        smoke="$root/smoke/${case_name}_${SLURM_ARRAY_JOB_ID:-manual}"
        "$python" -u "$code/run_official.py" run --upstream "$upstream" --inputs "$inputs" \
            --output "$smoke" --condition "$condition" --seed "$seed" --smoke
        "$python" -u "$code/run_official.py" fit --upstream "$upstream" --inputs "$inputs" \
            --snapshot "$smoke/fits/00200" --output "$smoke/peer" \
            --mask-function "$code/filter_swap.py:peer_aum_mask"
        ;;
    baseline)
        run_baseline
        ;;
    peer)
        run_peer
        ;;
    paired)
        run_baseline
        run_peer
        ;;
    *)
        echo "Expected smoke, baseline, peer, or paired" >&2
        exit 2
        ;;
esac
