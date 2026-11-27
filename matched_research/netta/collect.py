"""Build seed-level and setting-level paired results."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

CONDITIONS = ("clean", "sym30", "sym40", "sym50", "asym30", "asym40", "asym50")
SEEDS = (71, 72, 73)
BUDGETS = (400, 800, 1200, 1600, 2000)


def read(path):
    return json.loads(path.read_text())


def paired_rows(root):
    for condition in CONDITIONS:
        for seed in SEEDS:
            case = f"{condition}_s{seed}"
            reference_path = root / "runs" / case / "summary.json"
            if not reference_path.is_file():
                continue
            reference = read(reference_path)
            if (reference["condition"], reference["seed"], reference["smoke"]) != (condition, seed, False):
                raise ValueError(f"Wrong reference run in {reference_path}")
            by_budget = {cell["budget"]: cell for cell in reference["cells"]}
            for budget in BUDGETS:
                candidate_path = root / "peer" / case / f"{budget:05d}" / "result.json"
                if budget not in by_budget or not candidate_path.is_file():
                    continue
                base, candidate = by_budget[budget], read(candidate_path)
                if (candidate["condition"], candidate["seed"], candidate["budget"]) != (condition, seed, budget):
                    raise ValueError(f"Wrong candidate run in {candidate_path}")
                if base["kept"] != candidate["kept"]:
                    raise ValueError(f"Different training set sizes in {candidate_path}")
                if candidate["baseline_prediction_sha256"] != base["prediction_sha256"]:
                    raise ValueError(f"Reference predictions changed in {candidate_path}")
                identity = read(candidate_path.with_name("identity_result.json"))
                for field in ("prediction_sha256", "fitted_model_sha256"):
                    if identity[field] != base[field]:
                        raise ValueError(f"Identity replay mismatch in {case} at {budget} for {field}")
                if not math.isclose(identity["accuracy"], base["accuracy"], abs_tol=1e-5, rel_tol=0):
                    raise ValueError(f"Identity replay accuracy mismatch in {case} at {budget}")
                snapshot = reference_path.parent / "fits" / f"{budget:05d}" / "fit_inputs.npz"
                if hashlib.sha256(snapshot.read_bytes()).hexdigest() != candidate["fit_inputs_sha256"]:
                    raise ValueError(f"Fit input mismatch in {candidate_path}")
                if not all(0 <= record["accuracy"] <= 100 and math.isfinite(record["accuracy"])
                           for record in (base, candidate)):
                    raise ValueError(f"Invalid test accuracy in {candidate_path}")
                yield {
                    "condition": condition, "budget": budget, "seed": seed,
                    "base_accuracy_pct": base["accuracy"],
                    "peer_accuracy_pct": candidate["accuracy"],
                    "delta_pp": round(candidate["accuracy"] - base["accuracy"], 8),
                    "retained": base["kept"],
                    "replaced": candidate["changed_pairs"],
                    "base_clean_retained": candidate["baseline_retained_clean"],
                    "peer_clean_retained": candidate["retained_clean"],
                }


def setting_rows(pairs):
    for condition in CONDITIONS:
        for budget in BUDGETS:
            group = [x for x in pairs if (x["condition"], x["budget"]) == (condition, budget)]
            if group:
                yield {
                    "condition": condition, "budget": budget, "seeds": len(group),
                    "base_accuracy_pct": mean(row["base_accuracy_pct"] for row in group),
                    "peer_accuracy_pct": mean(row["peer_accuracy_pct"] for row in group),
                    "delta_pp": mean(row["delta_pp"] for row in group),
                    "winning_seeds": sum(row["delta_pp"] > 0 for row in group),
                    "tied_seeds": sum(row["delta_pp"] == 0 for row in group),
                }


def save_csv(path, rows):
    with path.open("x", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    pairs = list(paired_rows(args.root))
    total = len(CONDITIONS) * len(SEEDS) * len(BUDGETS)
    print(f"{len(pairs)}/{total} verified pairs")
    if not pairs or (len(pairs) != total and not args.partial):
        raise SystemExit("The requested paired grid is incomplete")
    settings = list(setting_rows(pairs))
    args.output.mkdir(parents=True, exist_ok=False)
    save_csv(args.output / "per_seed.csv", pairs)
    save_csv(args.output / "by_setting.csv", settings)
    complete = [row for row in settings if row["seeds"] == len(SEEDS)]
    wins = sum(row["delta_pp"] > 1e-8 for row in complete)
    ties = sum(abs(row["delta_pp"]) <= 1e-8 for row in complete)
    summary = {
        "matched_pairs": len(pairs), "planned_pairs": total,
        "complete_settings": len(complete), "wins": wins, "ties": ties,
        "mean_delta_pp": mean(row["delta_pp"] for row in complete) if complete else None,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
