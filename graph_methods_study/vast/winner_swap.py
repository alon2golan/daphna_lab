"""Use the compressed winner with the archived VAST posterior's sharpness."""

import argparse
import json
from pathlib import Path

import numpy as np


def swap_winners(original, proposed):
    rows = np.arange(len(original))
    old_winner = original.argmax(axis=1)
    new_winner = proposed.argmax(axis=1)
    revised = original.copy()
    revised[rows, new_winner] = original[rows, old_winner]
    revised[rows, old_winner] = original[rows, new_winner]
    return revised


def prepare(archive, compression, destination, name):
    from inspect_targets import target_stats
    from source_support_study.replay import DEPTH, checked_file, checksum, read
    from train_student import TARGET_FIELDS

    case = archive / "cells" / name
    original = case / f"A_{DEPTH}.npz"
    source = read(compression / "result.json")
    original_sha = read(case / "fixed_targets.json")["hashes"][original.name]
    if original_sha != source["original_sha256"] or source["depth"] != DEPTH:
        raise ValueError("Archived target does not match the compressed target")
    with np.load(checked_file(original, original_sha), allow_pickle=False) as saved:
        raw = {k: saved[k] for k in TARGET_FIELDS}
    with np.load(checked_file(compression / "targets.npz", source["revised_sha256"]),
                 allow_pickle=False) as saved:
        compressed = saved["probabilities"]
        for field in ("admitted", "confidence"):
            np.testing.assert_array_equal(saved[field], raw[field])

    revised = swap_winners(raw["probabilities"], compressed)
    np.testing.assert_array_equal(np.sort(revised, axis=1), np.sort(raw["probabilities"], axis=1))
    revised_prediction = revised.argmax(axis=1)
    proposed_winner = compressed.argmax(axis=1)
    old = raw["probabilities"].argmax(axis=1)
    gate = raw["admitted"]
    provenance = read(case / "inputs/provenance.json")
    truth_sha = read(archive / "plan.json")["features"]["train.labels"]
    truth = np.load(checked_file(Path(provenance["files"]["train.labels"]), truth_sha), allow_pickle=False)
    metrics = {
        "case": name, "depth": DEPTH, "admitted": int(gate.sum()), "original_sha256": original_sha,
        "compression_sha256": source["revised_sha256"],
        "changed_on_admitted": int((revised_prediction[gate] != old[gate]).sum()),
        "winner_ties_on_admitted": int((revised_prediction[gate] != proposed_winner[gate]).sum()),
        "raw": target_stats(raw["probabilities"][gate], truth[gate]),
        "compressed": target_stats(compressed[gate], truth[gate]),
        "winner_swap": target_stats(revised[gate], truth[gate]),
        "variant_source_sha256": checksum(Path(__file__)),
    }
    destination.mkdir(parents=True, exist_ok=True)
    targets_file = destination / "targets.npz"
    swap_path = destination / "targets.tmp"
    with swap_path.open("wb") as handle:
        np.savez_compressed(handle, probabilities=revised, admitted=gate, confidence=raw["confidence"])
    swap_path.replace(targets_file)
    metrics["revised_sha256"] = checksum(targets_file)
    summary = destination / "result.json"
    temporary = summary.with_suffix(".tmp")
    temporary.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    temporary.replace(summary)
    return metrics


def run(archive, compression, swap, output, index, check_only):
    from source_support_study.replay import checksum, read
    import train_student

    budget, condition = train_student.CASES[index]
    name = f"b{budget}_{condition}"
    screen = prepare(archive, compression / name, swap / name, name)
    train_student.train(archive, swap, output, index, check_only)
    if check_only:
        return
    destination = output / name
    result = read(destination / "comparison.json")
    if result["sqrt_target_sha256"] != screen["revised_sha256"]:
        raise ValueError("Fit used different targets")
    result["variant_target_sha256"] = result.pop("sqrt_target_sha256")
    result["winner_swap"] = result.pop("sqrt")
    result["variant_source_sha256"] = screen["variant_source_sha256"]
    result["screen_sha256"] = checksum(swap / name / "result.json")
    official = destination / "winner_comparison.json"
    path = official.with_suffix(".tmp")
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    path.replace(official)


if __name__ == "__main__":
    from train_student import CELLS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int, choices=CELLS)
    parser.add_argument("archive", type=Path)
    parser.add_argument("compression", type=Path)
    parser.add_argument("targets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run(args.archive, args.compression, args.targets, args.output, args.index, args.check_only)
