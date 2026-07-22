#!/bin/bash
# One command to launch the WHOLE coarse-budget sweep (all 21 budgets: 100..1000, 2k..10k, 25k,
# 50k) for several finetune VARIANTS, all in one NEW wandb group. Each variant is submitted as its
# OWN single job, so the variants run SIMULTANEOUSLY on separate GPUs. Because each variant is one
# job (not a shard array), it produces exactly TWO clean runs -- frozen and finetuned -- so the
# sidebar stays tidy: 2 runs per variant, split by curve and told apart by config.tag. No shards.
#
# Usage (no flags needed):
#   bash hierarchical_framework/scripts/launch_superclass_finetune.sh
#
# To start a fresh, separate sweep later, bump the GROUP suffix below (v2 -> v3 ...).

set -e

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
cd "$ROOT"                                   # submit from ROOT so the sbatch output paths resolve
JOB=hierarchical_framework/scripts/run_coarse_budget_sweep.sbatch   # single job (all budgets), 2 GPUs

GROUP=superclass_frozen_vs_finetuned_v2      # the NEW group; bump the suffix for a fresh sweep

# One sbatch per finetune variant -> they queue independently and run at the same time on separate
# GPUs. --tag names each; full default budgets (all 21) are used automatically (no --budgets).
sbatch "$JOB" --group "$GROUP" --tag ep30_lr1e4  --finetune_epochs 30  --lr_backbone 1e-4
sbatch "$JOB" --group "$GROUP" --tag ep80_lr3e5  --finetune_epochs 80  --lr_backbone 3e-5
sbatch "$JOB" --group "$GROUP" --tag ep100_lr1e4 --finetune_epochs 100 --lr_backbone 1e-4

echo "submitted 3 finetune variants (each its own job, running in parallel) to group '$GROUP'"
