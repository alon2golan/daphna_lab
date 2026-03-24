#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Collect local-sigma stats from experiment directories.")
    parser.add_argument(
        "--root",
        default="output",
        help="Root directory to scan for local_sigma_stats.json files (default: output)",
    )
    parser.add_argument(
        "--output",
        default="results_summaries/local_sigma_stats_summary.csv",
        help="CSV file to write (default: results_summaries/local_sigma_stats_summary.csv)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    json_paths = sorted(root.rglob("local_sigma_stats.json"))
    if not json_paths:
        raise SystemExit(f"No local_sigma_stats.json files found under {root}")

    rows = []
    for json_path in json_paths:
        with json_path.open("r", encoding="ascii") as handle:
            data = json.load(handle)
        data["stats_path"] = str(json_path)
        data["experiment_dir"] = str(json_path.parent)
        rows.append(data)

    fieldnames = [
        "dataset",
        "seed",
        "kernel_type",
        "local_sigma_k",
        "num_points",
        "mean",
        "std",
        "cv",
        "min",
        "q05",
        "q25",
        "median",
        "q75",
        "q95",
        "max",
        "iqr",
        "range",
        "experiment_dir",
        "stats_path",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
