#!/bin/bash
# ROUND TWO. Builds on the current winner ep80_lr3e5_lpft_no_bn (LP-FT: 15 head-only warmup
# epochs, then 80 unfrozen at lr 3e-5 with a cosine schedule + weight_decay 5e-4, BN left
# TRAINABLE -- freezing BN hurt because random selection makes each batch representative). Since
# the confirmed low-budget problem is feature DISTORTION, these variants attack it further:
#   finetune_blocks 1/2 -- only the last 1 or 2 ResNet stages move; earlier layers stay frozen,
#                          so distortion is confined to the most task-specific features.
#   label_smoothing 0.1 -- mild regularizer on the coarse cross-entropy.
#   lr 1e-5             -- an even gentler full-trunk finetune (less movement = less distortion).
#
# ADDITIVE: new --tag on each, so these are NEW wandb runs in the SAME existing group v2 -- the
# earlier runs, the earlier launchers, and every old command all still reproduce unchanged. All
# knobs default to a no-op, so re-running any prior sbatch gives the same result as before.
# Every variant is --skip_frozen (the frozen curve is already in v2).
#
# Usage (no flags):
#   bash hierarchical_framework/scripts/launch_superclass_finetune_partial.sh

set -e

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
cd "$ROOT"                                   # submit from ROOT so the sbatch output paths resolve
JOB=hierarchical_framework/scripts/run_coarse_budget_sweep.sbatch   # single job (all budgets), 2 GPUs

GROUP=superclass_frozen_vs_finetuned_v2      # the EXISTING group -> new lines land on the same panel

# The LP-FT base that won round one (BN left trainable). Each variant appends ONE new knob so you
# can read its effect off the panel in isolation.
LPFT="--finetune_epochs 80 --lr_backbone 3e-5 --skip_frozen \
  --head_warmup_epochs 15 --lr_schedule cosine --lr_warmup_epochs 5 --weight_decay 5e-4"

# Primary bet: finetune only the last ResNet block (layer4). Most task-specific layer moves; the
# rest of the pretrained trunk is frozen, so there is little to distort at low budget.
sbatch "$JOB" --group "$GROUP" --tag ep80_lr3e5_lpft_blk1 $LPFT --finetune_blocks 1

# Slightly more capacity: unfreeze the last two stages (layer3 + layer4).
sbatch "$JOB" --group "$GROUP" --tag ep80_lr3e5_lpft_blk2 $LPFT --finetune_blocks 2

# Full trunk, add label smoothing -- an orthogonal, near-free regularizer.
sbatch "$JOB" --group "$GROUP" --tag ep80_lr3e5_lpft_ls $LPFT --label_smoothing 0.1

# Full trunk, even gentler backbone lr (1e-5 instead of 3e-5) -- less movement, less distortion.
sbatch "$JOB" --group "$GROUP" --tag ep80_lr1e5_lpft \
  --finetune_epochs 80 --lr_backbone 1e-5 --skip_frozen \
  --head_warmup_epochs 15 --lr_schedule cosine --lr_warmup_epochs 5 --weight_decay 5e-4

echo "submitted 4 round-two variants (skip_frozen) to existing group '$GROUP'"
