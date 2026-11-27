"""Export paired protected VAST student results and the archived swap pilot."""

import argparse
import csv
import json
from pathlib import Path

from rank import CASES, DEPTHS


def collect(verified, swap_archive):
    for case in CASES:
        path = verified / case / "comparison.json"
        result = json.loads(path.read_text())
        budget, noise = case[1:].split("_", 1)
        if (result["case"] != case or result["labeled"] != int(budget)
                or result["test_count"] != 10000 or result["epochs"] != 300
                or result["seed"] != 0 or result["permutation"] not in DEPTHS):
            raise ValueError(f"Unexpected VAST evaluation in {path}")
        reference = result["archive"]["test_accuracy_percent"]
        candidate = result["candidate"]["test_accuracy_percent"]
        prior_swap = swap_archive / f"{case}.json"
        swap = None
        if prior_swap.is_file():
            record = json.loads(prior_swap.read_text())
            if record["archived_reference"] != result["archive"]:
                raise ValueError(f"Prior winner swap has a different reference in {case}")
            swap = round(record["winner_swap"]["test_accuracy_percent"], 2)
        yield {
            "condition": noise, "budget": int(budget), "seed": result["seed"],
            "ranks_aligned": result["permutation"], "admitted": result["admitted"],
            "optimizer_steps": result["optimizer_steps"],
            "reference_pct": round(reference, 2), "candidate_pct": round(candidate, 2),
            "delta_pp": round(candidate - reference, 2), "prior_winner_swap_pct": swap,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verified", type=Path)
    parser.add_argument("swap_archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = list(collect(args.verified, args.swap_archive))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
