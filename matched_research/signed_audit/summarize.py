"""Summarize signed verification against every matched arm and their per-cell envelope."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean, stdev


SIGNED = "signed_peer"
NO_REQUEST = "no_audit"
METRICS = {"graph": "test_correct", "cosine_1nn": "test_1nn_correct"}


def summarize(metrics_file):
    settings = defaultdict(lambda: defaultdict(dict))
    with metrics_file.open(newline="") as stream:
        for row in csv.DictReader(stream):
            condition = row["condition"]
            budget, seed = int(row["budget"]), int(row["seed"])
            policies = settings[condition, budget][seed]
            if row["policy"] in policies:
                raise ValueError("Duplicate evaluation")
            if int(row["test_count"]) != 10_000:
                raise ValueError("Expected the frozen 10,000-row held-out set")
            policies[row["policy"]] = row

    for (condition, budget), seeds in sorted(settings.items()):
        arm_names = set(seeds[min(seeds)])
        if any(set(rows) != arm_names for rows in seeds.values()):
            raise ValueError("Incomplete matched arms")
        if set(seeds) != set(range(80, 85)):
            raise ValueError("Incomplete seed roster")
        methods = sorted(arm_names - {SIGNED})
        controls = [arm for arm in methods if arm != NO_REQUEST]
        for metric, column in METRICS.items():
            correct = {
                policy: [int(seeds[seed][policy][column]) for seed in sorted(seeds)]
                for policy in arm_names
            }
            best = max(controls, key=lambda arm: sum(correct[arm]))
            for baseline in [*methods, "best_equal_cost_mean"]:
                reference = best if baseline == "best_equal_cost_mean" else baseline
                changes = [a - b for a, b in zip(correct[SIGNED], correct[reference])]
                yield {
                    "condition": condition, "budget": budget, "metric": metric,
                    "baseline": baseline, "reference_arm": reference,
                    "signed_pct": round(mean(correct[SIGNED]) / 100, 5),
                    "baseline_pct": round(mean(correct[reference]) / 100, 5),
                    "delta_pp": round(mean(changes) / 100, 5),
                    "paired_sd_pp": round(stdev(changes) / 100, 5),
                    "wins": sum(change > 0 for change in changes),
                    "ties": sum(change == 0 for change in changes),
                    "losses": sum(change < 0 for change in changes),
                    "seeds": len(seeds),
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    rows = list(summarize(args.metrics_csv))
    if len(rows) != 512:
        raise ValueError(f"Expected 512 complete setting comparisons, found {len(rows)}")
    with args.output_csv.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
