#!/bin/bash
# Adds LOW-BUDGET-STABILIZED finetune variants to the EXISTING wandb group superclass_frozen_vs_
# finetuned_v2 (same panel as the first sweep). All start from the current winner (80 epochs, lr
# 3e-5) and layer on the three fixes for why finetuning lost to the frozen probe below ~3k:
#   freeze_bn           -- BatchNorm stays in eval mode, so its running stats are the clean
#                          pretrained ones instead of being re-estimated on a ~100-point batch.
#   head_warmup_epochs  -- LP-FT: train the head only for 15 epochs before unfreezing, so the
#                          random head stops distorting the pretrained trunk on step one.
#   lr_schedule cosine  -- 5-epoch backbone-lr ramp then cosine decay to ~0 (vs a flat lr).
#
# Every variant is --skip_frozen because the frozen curve is ALREADY in v2 from the first sweep;
# these jobs only add new finetuned lines. Each is its own single job (2 clean runs -> here just
# the one finetuned run) so they run in parallel and the sidebar stays tidy.
#
# Usage (no flags):
#   bash hierarchical_framework/scripts/launch_superclass_finetune_stabilizers.sh

set -e

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
cd "$ROOT"                                   # submit from ROOT so the sbatch output paths resolve
JOB=hierarchical_framework/scripts/run_coarse_budget_sweep.sbatch   # single job (all budgets), 2 GPUs

GROUP=superclass_frozen_vs_finetuned_v2      # the EXISTING group -> new lines land on the same panel

# All three build on the 80ep/3e-5 winner and add --skip_frozen (frozen already plotted in v2).
# bn_only isolates the BatchNorm fix; lpft_bn stacks LP-FT + cosine + stronger weight decay on
# top of it (my primary recommendation); lpft_no_bn is the ablation with the BN fix removed so
# you can read off how much each piece contributes.
sbatch "$JOB" --group "$GROUP" --tag ep80_lr3e5_bn \
  --finetune_epochs 80 --lr_backbone 3e-5 --skip_frozen \
  --freeze_bn

sbatch "$JOB" --group "$GROUP" --tag ep80_lr3e5_lpft_bn \
  --finetune_epochs 80 --lr_backbone 3e-5 --skip_frozen \
  --freeze_bn --head_warmup_epochs 15 --lr_schedule cosine --lr_warmup_epochs 5 --weight_decay 5e-4

sbatch "$JOB" --group "$GROUP" --tag ep80_lr3e5_lpft_no_bn \
  --finetune_epochs 80 --lr_backbone 3e-5 --skip_frozen \
  --head_warmup_epochs 15 --lr_schedule cosine --lr_warmup_epochs 5 --weight_decay 5e-4

echo "submitted 3 stabilized finetune variants (skip_frozen) to existing group '$GROUP'"
