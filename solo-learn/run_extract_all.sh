#!/usr/bin/env bash
# Submit embedding extraction for all four final (1000ep) runs at once:
# simclr/byol x cifar10/cifar100. Each is a separate GPU sbatch job, so they run
# in parallel. Each writes train/test_features.npy (+ labels) into
# embeddings/<method>/<dataset>/1000ep/ via the fixed extract_embeddings.py
# (deterministic T_val transform, no shuffle, no drop_last -> real images, full
# 50000 train / 10000 test, in order).
#
# Usage (from the solo-learn folder):  ./run_extract_all.sh

set -e

# run dir + dataset name for each of the four final runs
RUNS=(
  "trained_models/simclr/cifar10/1000ep   cifar10"
  "trained_models/simclr/cifar100/1000ep  cifar100"
  "trained_models/byol/cifar10/1000ep     cifar10"
  "trained_models/byol/cifar100/1000ep    cifar100"
)

for entry in "${RUNS[@]}"; do
  # shellcheck disable=SC2086 -- deliberately split into ckpt_dir + dataset
  set -- $entry
  ckpt_dir=$1
  dataset=$2
  echo "submitting: $ckpt_dir ($dataset)"
  sbatch run_extract.sbatch "$ckpt_dir" "$dataset"
done

echo "submitted ${#RUNS[@]} jobs -- watch with: squeue -u \$USER"
