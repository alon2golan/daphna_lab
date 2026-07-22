# Optional post-step for the PARALLEL (job-array) sweep: the live view is already the overlaid
# wandb group, so this is only for a single tidy artifact. It reads every shard CSV for a
# (selector, seed), concatenates + sorts them by budget, and writes ONE combined CSV + PNG under
# outputs/finetune/ (and, with --wandb, one consolidated 'merged' run holding the full curves).

import argparse
import csv
import sys
from pathlib import Path

# pipeline/ on the path (for the shared coarse map, though we mainly reuse the sibling runner)
FINETUNE_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = FINETUNE_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

# Reuse the runner's disk/plot/wandb helpers so the merged artifact is identical in style
from shared.superclass_maps import CIFAR100_COARSE
from two_head_finetune import FINETUNE_EPOCHS, LR_BACKBONE
from run_coarse_budget_sweep import (
    RESULTS_DIR, CSV_HEADER, save_figure, init_curve_run,
    DEFAULT_WANDB_GROUP, wandb,
)


# --- Read the shard CSVs ---

def load_shards(selector, seed, tag=""):
    """Reads every per-shard CSV for this (selector, seed, tag) and returns rows sorted by
    num_labeled. De-duplicates by num_labeled (last one wins) in case a shard was re-run.
    `tag` selects one finetune variant so variants don't get mixed together."""
    tag_suffix = f"_{tag}" if tag else ""                               # match this variant only
    pattern = f"coarse_budget_sweep_{selector}_seed{seed}{tag_suffix}_shard*.csv"   # its shard files
    by_budget = {}                                                       # num_labeled -> parsed row
    for csv_file in sorted(RESULTS_DIR.glob(pattern)):
        with open(csv_file) as handle:
            for row in csv.DictReader(handle):
                by_budget[int(row["num_labeled"])] = row                 # keyed so re-runs overwrite
    ordered = [by_budget[k] for k in sorted(by_budget)]                  # ascending budget
    return ordered


# --- Write the combined artifacts ---

def write_combined_csv(rows, selector, seed, tag=""):
    """Writes the merged rows (already the runner's column order) into one clean CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{tag}" if tag else ""
    csv_path = RESULTS_DIR / f"coarse_budget_sweep_{selector}_seed{seed}{tag_suffix}.csv"
    with open(csv_path, "w") as handle:
        handle.write(CSV_HEADER + "\n")
        for row in rows:
            handle.write(",".join(row[col] for col in CSV_HEADER.split(",")) + "\n")
    print(f"saved -> {csv_path}", flush=True)
    return csv_path


def to_results(rows):
    """Maps the parsed CSV rows into the {num_labeled, frozen_acc, finetuned_acc} the plot wants."""
    return [{"num_labeled": int(r["num_labeled"]),
             "frozen_acc": float(r["frozen_acc"]),
             "finetuned_acc": float(r["finetuned_acc"])} for r in rows]


# --- Orchestration ---

def main():
    parser = argparse.ArgumentParser(
        description="Merge the parallel coarse-budget shard CSVs into one CSV + PNG (+ optional wandb).")
    parser.add_argument("--selector", default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true",
                        help="also log one consolidated 'merged' run holding the full curves")
    parser.add_argument("--project", default="superclass_analysis")
    parser.add_argument("--group", default=DEFAULT_WANDB_GROUP)
    parser.add_argument("--tag", default="", help="which finetune variant to merge (matches --tag "
                        "from the sweep); omit for an untagged run")
    args = parser.parse_args()

    rows = load_shards(args.selector, args.seed, tag=args.tag)
    if not rows:
        raise SystemExit(f"no shard CSVs found for selector={args.selector} seed={args.seed} "
                         f"tag={args.tag or '(none)'} in {RESULTS_DIR}")
    print(f"merging {len(rows)} budgets for selector={args.selector} seed={args.seed} "
          f"tag={args.tag or '(none)'}", flush=True)

    write_combined_csv(rows, args.selector, args.seed, tag=args.tag)
    results = to_results(rows)
    png_path = save_figure(results, args.selector, args.seed, tag=args.tag)   # combined curve PNG

    # Optional: two clean consolidated runs (frozen + finetuned) each holding the WHOLE curve,
    # separate from the live per-shard runs. Same metric key -> two lines when grouped by curve.
    if args.wandb and wandb is not None:
        num_coarse = int(CIFAR100_COARSE.max()) + 1                    # 20 superclasses
        for curve, key in (("frozen", "frozen_acc"), ("finetuned", "finetuned_acc")):
            run = init_curve_run(args.project, args.group, curve, args.selector, args.seed,
                                 embeddings_tag="merged", num_coarse=num_coarse, shard="merged",
                                 epochs=FINETUNE_EPOCHS, lr_backbone=LR_BACKBONE, tag=args.tag)
            for entry in results:
                run.log({"num_labeled": entry["num_labeled"], "coarse_acc": entry[key]})
            run.log({"comparison_figure": wandb.Image(str(png_path))})
            run.finish()


if __name__ == "__main__":
    main()
