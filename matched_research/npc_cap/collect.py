"""Collect every paired official, raw-graph and capped-graph test score."""

import argparse
import csv
from pathlib import Path
from statistics import mean

from replay import BUDGETS, CASES, file_sha256, read


def rows_from(source):
    graph_sha = file_sha256(source / "geometry.npz")
    replay_sha = file_sha256(Path(__file__).with_name("replay.py"))
    cap_source = Path(__file__).resolve().parents[2] / "graph_methods_study/sota/capped_votes/capped_votes.py"
    cap_sha = file_sha256(cap_source)
    for condition, seed in CASES:
        artifact = read(source / f"{condition}_s{seed}.json")
        if ((artifact["condition"], artifact["seed"], artifact["test_count"]) != (condition, seed, 10000)
                or artifact["graph_sha256"] != graph_sha or artifact["source_sha256"] != replay_sha
                or artifact["cap_sha256"] != cap_sha
                or tuple(item["budget"] for item in artifact["records"]) != BUDGETS):
            raise ValueError(f"Mismatched graph replay in {condition} seed {seed}")
        for item in artifact["records"]:
            official, raw, capped = [item[field] for field in ("official", "raw", "cap")]
            if not all(isinstance(n, int) and 0 <= n <= 10000 for n in (official, raw, capped)):
                raise ValueError(f"Invalid test counts in {condition} seed {seed}")
            yield dict(
                condition=condition, budget=item["budget"], seed=seed, retained=item["retained"],
                official_pct=official / 100, raw_pct=raw / 100, cap_pct=capped / 100,
                delta_vs_official_pp=(capped - official) / 100, delta_cap_pp=(capped - raw) / 100,
                delta_raw_pp=(raw - official) / 100,
            )


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verified", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    per_seed = list(rows_from(args.verified))
    settings = {}
    for row in per_seed:
        settings.setdefault((row["condition"], row["budget"]), []).append(row)
    fields = ("official_pct", "raw_pct", "cap_pct", "delta_vs_official_pp", "delta_cap_pp", "delta_raw_pp")
    by_setting = [
        dict(condition=condition, budget=budget, seeds=len(trials),
             **{field: mean(t[field] for t in trials) for field in fields})
        for (condition, budget), trials in settings.items()
    ]
    args.output.mkdir(parents=True)
    write_csv(args.output / "per_seed.csv", per_seed)
    write_csv(args.output / "by_setting.csv", by_setting)
    print(f"Verified {len(per_seed)} paired scores across {len(by_setting)} settings")


if __name__ == "__main__":
    main()
