"""
Merge all partial parquet files produced by sweep_generate_lsets.py into the
final *_all.parquet files consumed by get_train_idx().

Run this AFTER all SLURM array jobs have completed:
    python batch_utils/merge_lsets_parquets.py

Output paths (one per AL method):
    output/CIFAR100/cifar100_resnet18_probcover_sym_all.parquet
    output/CIFAR100/cifar100_resnet18_maxherding_sym_all.parquet
"""

import glob
import os
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration — must match generate_lsets_parquet_worker.py and the yaml
# ---------------------------------------------------------------------------
DATASET_UPPER = "CIFAR100"
DATASET_LOWER = "cifar100"
MODEL         = "resnet18"
NOISE_TYPE    = "sym"
AL_METHODS    = ["probcover", "maxherding"]

OUTPUT_DIR    = Path(_repo_root) / "output" / DATASET_UPPER
PARTIAL_DIR   = OUTPUT_DIR / "partial"


def merge(al_method: str, dry_run: bool = False) -> None:
    pattern = str(
        PARTIAL_DIR / f"{DATASET_LOWER}_{MODEL}_{al_method}_{NOISE_TYPE}_*.parquet"
    )
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[{al_method}] No partial files found matching: {pattern}")
        return

    out_path = OUTPUT_DIR / f"{DATASET_LOWER}_{MODEL}_{al_method}_{NOISE_TYPE}_all.parquet"

    if dry_run:
        print(f"[{al_method}] Would merge {len(files)} partial files → {out_path}")
        for f in files:
            print(f"    {f}")
        return

    print(f"[{al_method}] Reading {len(files)} partial files …")
    dfs = [pd.read_parquet(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)

    # Sanity-check schema
    required_cols = {"budget", "oracle_teacher_ratio", "alpha", "lsets"}
    missing = required_cols - set(merged.columns)
    if missing:
        raise ValueError(f"Merged DataFrame is missing columns: {missing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    n_oracle = (merged["alpha"] == 1).sum()
    n_noisy  = (merged["alpha"] == 0).sum()
    print(
        f"[{al_method}] Merged {len(files)} files, "
        f"{len(merged)} total rows "
        f"(oracle={n_oracle}, noisy={n_noisy}) → {out_path}"
    )


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Merge partial lsets parquet files")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be merged without writing")
    ap.add_argument("--al", nargs="*", default=AL_METHODS,
                    help=f"AL methods to merge (default: {AL_METHODS})")
    args = ap.parse_args()

    for al_method in args.al:
        merge(al_method, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
