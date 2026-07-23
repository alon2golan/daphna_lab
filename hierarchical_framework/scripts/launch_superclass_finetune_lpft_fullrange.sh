#!/bin/bash
# Extends the 3-seed LP-FT winner (tag lpft_base) from <=3k out to the FULL budget range, so its
# frozen and finetuned lines run all the way to 50k at 3 seeds -- matching the single-seed originals.
#
# CONTINUES THE SAME LINES (no new lines): these jobs reuse the EXACT tag lpft_base and seeds 0/1/2
# and run ONLY the budgets not already done (4k..50k). Because the panel groups by config.curve +
# config.tag, the new high-budget points aggregate onto the existing lpft_base frozen + finetuned
# lines. New run objects appear in the sidebar but collapse under the group -> one continuous line
# per curve. Rows also append to the same per-seed CSVs (coarse_budget_sweep_random_seed*_lpft_base
# .csv), so aggregate_seeds.py sees the full range.
#
# NOT --skip_frozen: the frozen benchmark is also only computed to 3k so far, so we let lpft_base
# compute it for the new budgets too -> the grey benchmark line extends to 50k at 3 seeds as well.
#
# Usage (no flags):
#   bash hierarchical_framework/scripts/launch_superclass_finetune_lpft_fullrange.sh

set -e

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
cd "$ROOT"                                    # submit from ROOT so the sbatch output paths resolve
JOB=hierarchical_framework/scripts/run_coarse_budget_sweep.sbatch   # single job, 2 GPUs

GROUP=superclass_frozen_vs_finetuned_v2       # the EXISTING group -> same panel
HI_BUDGETS="--budgets 4000 5000 6000 7000 8000 9000 10000 25000 50000"  # only the ones not yet run

# EXACT round-one LP-FT recipe behind tag lpft_base (full trunk, BN trainable). Must match the
# earlier lpft_base jobs so the extended points belong to the same variant.
BASE="--finetune_epochs 80 --lr_backbone 3e-5 --head_warmup_epochs 15 \
  --lr_schedule cosine --lr_warmup_epochs 5 --weight_decay 5e-4"

for SEED in 0 1 2; do
  sbatch "$JOB" --group "$GROUP" --seed $SEED $HI_BUDGETS --tag lpft_base $BASE
done

echo "submitted lpft_base high-budget extension (4k..50k) x seeds 0/1/2 to group '$GROUP'"
echo "the 50k finetune is the long pole; jobs are --killable --requeue"
