#!/bin/bash
# BUDGET-PARALLEL variant of the launcher: each finetune variant runs its budgets ACROSS GPUs
# (a 6-shard job array), then a dependent merge builds the CLEAN ordered runs. Trade-off vs the
# simple launcher: this is faster (budgets run in parallel) but the clean per-variant runs appear
# when the merge finishes, NOT live -- because separate GPU tasks can't safely share one live
# wandb run. During the run the shards write CSVs only (no wandb clutter); the merge then logs
# exactly one frozen + one finetuned run per variant, with all 21 budgets in order.
#
# Result in the group: one frozen run + three finetuned runs (frozen computed once), each a single
# connected line -- same tidy sidebar as the simple launcher, produced at the end.
#
# Usage (no flags needed):
#   bash hierarchical_framework/scripts/launch_superclass_finetune_parallel.sh

set -e

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
cd "$ROOT"                                   # submit from ROOT so the sbatch output paths resolve
ARRAY=hierarchical_framework/scripts/run_coarse_budget_sweep_array.sbatch
MERGE=hierarchical_framework/scripts/run_coarse_budget_merge.sbatch

GROUP=superclass_frozen_vs_finetuned_v2      # the NEW group; bump the suffix for a fresh sweep

# submit_variant TAG EPOCHS LR EXTRA...  -> array (budgets parallel, CSV only) + dependent merge (wandb)
submit_variant() {
  local tag=$1 epochs=$2 lr=$3; shift 3
  # Array with no --wandb -> shards just write CSVs across GPUs in parallel
  local aid
  aid=$(sbatch --parsable "$ARRAY" --group "$GROUP" --tag "$tag" \
               --finetune_epochs "$epochs" --lr_backbone "$lr" "$@")
  # Merge runs only after all shards succeed; it logs the clean frozen/finetuned runs + PNG
  sbatch --dependency=afterok:"$aid" "$MERGE" --group "$GROUP" --tag "$tag" --wandb
  echo "variant $tag: array=$aid -> merge queued (afterok)"
}

# Frozen is identical across variants -> compute it in the first variant only; the rest --skip_frozen
submit_variant ep30_lr1e4  30  1e-4
submit_variant ep80_lr3e5  80  3e-5 --skip_frozen
submit_variant ep100_lr1e4 100 1e-4 --skip_frozen

echo "submitted 3 budget-parallel variants to group '$GROUP' (clean runs appear after each merge)"
