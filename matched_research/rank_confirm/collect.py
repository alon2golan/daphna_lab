"""Validate completed trajectories and write the paired confirmation tables."""

import argparse
import csv
from pathlib import Path
from statistics import mean

from run_fresh import CASES, GRAPH_SHA, WRAPPER_SHA
from run_official import BUDGETS, file_sha256, write_json
from score import RAW_DEPTHS, RANK_SHA, read


DEPTH_COLUMNS = {f"raw{depth}": f"raw{depth}_pct" if depth != 6 else "raw_pct" for depth in RAW_DEPTHS}
PERCENT_COLUMNS = {"official": "official_pct", "rank6": "rank_pct", **DEPTH_COLUMNS}
DELTA_COLUMNS = ("delta_rank_pp", "delta_rank_official_pp")


def collect(archive, output):
    scorer_sha = file_sha256(Path(__file__).with_name("score.py"))
    parity = read(archive / "parity.json")
    if ((parity["condition"], parity["seed"], parity["source_sha256"]) != ("clean", 71, scorer_sha)
            or tuple(row["budget"] for row in parity["rows"]) != BUDGETS):
        raise ValueError("Missing archived score parity")
    pairs, settings, artifacts = [], {}, {}
    for condition, seed in CASES:
        case = f"{condition}_s{seed}"
        scored = archive / "results" / f"{case}.json"
        stock = archive / "runs" / case / "summary.json"
        stock_provenance = stock.with_name("provenance.json")
        data, stock_rows = read(scored), read(stock)["cells"]
        if ((data["condition"], data["seed"], data["test_count"]) != (condition, seed, 10000)
                or data["source_sha256"] != scorer_sha
                or data["wrapper_sha256"] != WRAPPER_SHA
                or data["graph_sha256"] != GRAPH_SHA
                or data["rank_sha256"] != RANK_SHA
                or data["stock_summary_sha256"] != file_sha256(stock)
                or data["stock_provenance_sha256"] != file_sha256(stock_provenance)
                or tuple(row["budget"] for row in data["rows"]) != BUDGETS):
            raise ValueError(f"Mismatched score inputs for {case}")
        artifacts[case] = {
            "scores_sha256": file_sha256(scored),
            "stock_summary_sha256": data["stock_summary_sha256"],
            "stock_provenance_sha256": data["stock_provenance_sha256"],
        }
        for row, official in zip(data["rows"], stock_rows, strict=True):
            budget = row["budget"]
            if (row["official"] / 100 != official["accuracy"]
                    or row["input_sha256"] != official["fit_inputs_sha256"]
                    or row["official_prediction_sha256"] != official["prediction_sha256"]
                    or budget != official["budget"] or row["retained"] != official["kept"]
                    or any(type(row[key]) is not int or not 0 <= row[key] <= 10000 for key in PERCENT_COLUMNS)):
                raise ValueError(f"Unpaired scores for {case} at {budget} labels")
            record = dict(condition=condition, budget=budget, seed=seed, retained=row["retained"],
                          **{column: row[key] / 100 for key, column in PERCENT_COLUMNS.items()},
                          delta_rank_pp=(row["rank6"] - row["raw6"]) / 100,
                          delta_rank_official_pp=(row["rank6"] - row["official"]) / 100)
            pairs.append(record)
            settings.setdefault((condition, budget), []).append(record)
    averages = [dict(condition=condition, budget=budget, seeds=len(rows),
                     **{column: mean(row[column] for row in rows)
                        for column in (*PERCENT_COLUMNS.values(), *DELTA_COLUMNS)})
                for (condition, budget), rows in settings.items()]
    output.mkdir(parents=True)
    for name, rows in (("per_seed.csv", pairs), ("by_setting.csv", averages)):
        with (output / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    write_json(output / "provenance.json", {
        "cases": artifacts, "parity_sha256": file_sha256(archive / "parity.json"),
        "score_source_sha256": scorer_sha, "graph_sha256": GRAPH_SHA, "wrapper_sha256": WRAPPER_SHA,
    })
    print(f"Verified {len(pairs)} pairs over {len(averages)} settings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    collect(args.archive, args.output)
