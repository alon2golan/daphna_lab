"""Measure whether class probabilities improve along with teacher argmax labels."""

import argparse
import json
from pathlib import Path

import numpy as np

from source_support_study.replay import DEPTH, checked_file, checksum, read
from transfer import CASES


def target_stats(p, truth):
    p = p.astype(np.float64)
    actual = p[np.arange(len(p)), truth]
    return {
        "entropy_nats": float((-p * np.log(p)).sum(axis=1).mean()),
        "mean_true_class_probability": float(actual.mean()),
        "negative_log_likelihood": float(-np.log(actual).mean()),
        "mean_top_probability": float(p.max(axis=1).mean()),
        "correct": int((p.argmax(axis=1) == truth).sum()),
    }


def inspect(archive, targets, output):
    plan = read(archive / "plan.json")
    cases = []
    for budget, condition in CASES:
        name = f"b{budget}_{condition}"
        original = archive / "cells" / name
        target = targets / name
        reference = read(target / "result.json")
        provenance = read(original / "inputs/provenance.json")
        truth_path = checked_file(Path(provenance["files"]["train.labels"]), plan["features"]["train.labels"])
        truth = np.load(truth_path, allow_pickle=False)
        paths = (checked_file(original / f"A_{DEPTH}.npz", reference["original_sha256"]),
                 checked_file(target / "targets.npz", reference["revised_sha256"]))
        arms = {}
        for arm, path in zip(("raw", "sqrt"), paths, strict=True):
            with np.load(path, allow_pickle=False) as saved:
                gate = saved["admitted"]
                stats = target_stats(saved["probabilities"][gate], truth[gate])
                if stats["correct"] != reference["rows"]["admitted"][arm]:
                    raise ValueError(f"Admitted count failed replay in {name}")
                arms[arm] = stats
        result = {"case": name, "budget": budget, "condition": condition, "admitted": int(gate.sum()),
                  "original_sha256": reference["original_sha256"], "sqrt_sha256": reference["revised_sha256"],
                  "arms": arms}
        cases.append(result)
        print(json.dumps(result), flush=True)
    with output.open("x") as handle:
        json.dump({"source_sha256": checksum(Path(__file__)), "cases": cases}, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("targets", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    inspect(args.archive, args.targets, args.output)
