#!/bin/bash
# One command to launch the WHOLE coarse-budget sweep (all 21 budgets: 100..1000, 2k..10k, 25k,
# 50k) for several finetune VARIANTS at once, all in one NEW wandb group. Each variant is its own
# job array (6 shards, one GPU each, live). Frozen and finetuned show up as separate runs; the
# variants are told apart by config.tag.
#
# Usage (no flags needed):
#   bash hierarchical_framework/scripts/launch_superclass_finetune.sh
#
# To start a fresh, separate sweep later, bump the GROUP suffix below (v2 -> v3 ...).

set -e

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
cd "$ROOT"                                   # submit from ROOT so the sbatch output paths resolve
ARRAY=hierarchical_framework/scripts/run_coarse_budget_sweep_array.sbatch

GROUP=superclass_frozen_vs_finetuned_v2      # the NEW group; bump the suffix for a fresh sweep

# One line per finetune variant: --tag names it, plus the technique knobs. Full default budgets
# are used automatically (no --budgets), so every variant covers all 21 budgets.
sbatch "$ARRAY" --group "$GROUP" --tag ep30_lr1e4  --finetune_epochs 30  --lr_backbone 1e-4
sbatch "$ARRAY" --group "$GROUP" --tag ep80_lr3e5  --finetune_epochs 80  --lr_backbone 3e-5
sbatch "$ARRAY" --group "$GROUP" --tag ep100_lr1e4 --finetune_epochs 100 --lr_backbone 1e-4

echo "submitted 3 finetune variants to wandb group '$GROUP' (all 21 budgets, 6 shards each)"
