#!/bin/bash
# ROUND THREE: the two finetune-ONLY levers (they move only the finetuned embedding, so the frozen
# curve never changes and never needs rerunning), plus a real multi-SEED confirmation because the
# round-two variants differed by <=0.5 at seed 0 -- inside the noise. Restricted to the <=3k budget
# region you care about, so 12 jobs stay cheap.
#
#   aug=strong  -- adds SimCLR color jitter + grayscale to the finetune augmentation (the default
#                  already crops+flips); the biggest untapped lever at tiny budgets.
#   l2sp=1e-3   -- L2-SP: pull backbone weights toward their PRETRAINED values instead of toward
#                  zero -- an explicit anti-distortion regularizer, the mechanism we confirmed.
#   stack       -- blk1 (layer4 only) + label smoothing + a longer 40-epoch LP warmup + strong aug.
#
# Base recipe = the round-one LP-FT winner: 15 head-only warmup epochs, then 80 unfrozen at lr 3e-5
# with a cosine schedule + weight_decay 5e-4, BN left trainable. Run at seeds 0,1,2.
#
# ADDITIVE: new --tag per variant, seed in the run name, all into the EXISTING group v2. The base
# variant computes the frozen curve per seed (a fair reference for that seed); the others are
# --skip_frozen. Nothing old is touched; all knobs default to no-ops so every prior command still
# reproduces. Aggregate the per-seed CSVs afterward for mean +/- std.
#
# Usage (no flags):
#   bash hierarchical_framework/scripts/launch_superclass_finetune_round3.sh

set -e

ROOT=/cs/labs/daphna/$USER/projects/daphna_lab
cd "$ROOT"                                   # submit from ROOT so the sbatch output paths resolve
JOB=hierarchical_framework/scripts/run_coarse_budget_sweep.sbatch   # single job, 2 GPUs

GROUP=superclass_frozen_vs_finetuned_v2      # the EXISTING group -> new lines land on the same panel
BUDGETS="--budgets 100 200 300 400 500 600 700 800 900 1000 2000 3000"   # the <=3k region only

# The round-one LP-FT base (BN trainable). Each variant appends ONE lever so its effect is isolated.
BASE="--finetune_epochs 80 --lr_backbone 3e-5 --head_warmup_epochs 15 \
  --lr_schedule cosine --lr_warmup_epochs 5 --weight_decay 5e-4"

for SEED in 0 1 2; do
  # Reference finetune + the frozen curve for THIS seed (no --skip_frozen here only).
  sbatch "$JOB" --group "$GROUP" --seed $SEED $BUDGETS --tag lpft_base $BASE

  # Lever 1: strong augmentation.
  sbatch "$JOB" --group "$GROUP" --seed $SEED $BUDGETS --tag lpft_aug $BASE --skip_frozen --aug strong

  # Lever 2: L2-SP anchor to pretrained weights (lower the plain weight_decay since L2-SP now
  # supplies the backbone regularization; keep a little wd for the heads).
  sbatch "$JOB" --group "$GROUP" --seed $SEED $BUDGETS --tag lpft_l2sp \
    --finetune_epochs 80 --lr_backbone 3e-5 --head_warmup_epochs 15 \
    --lr_schedule cosine --lr_warmup_epochs 5 --weight_decay 1e-4 --skip_frozen --l2sp 1e-3

  # Stack: partial finetune (layer4 only) + label smoothing + longer LP warmup + strong aug.
  sbatch "$JOB" --group "$GROUP" --seed $SEED $BUDGETS --tag lpft_blk1_ls_aug \
    --finetune_epochs 80 --lr_backbone 3e-5 --head_warmup_epochs 40 \
    --lr_schedule cosine --lr_warmup_epochs 5 --weight_decay 5e-4 --skip_frozen \
    --finetune_blocks 1 --label_smoothing 0.1 --aug strong
done

echo "submitted 4 variants x 3 seeds = 12 jobs (<=3k budgets) to existing group '$GROUP'"
