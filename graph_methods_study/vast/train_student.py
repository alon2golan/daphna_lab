"""Fit the archived protected VAST student with square-root evidence targets."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from source_support_study.replay import DEPTH, checked_file, checksum, read
from transfer import CASES

CELLS = (0, 1, 2, 3, 5, 6)
TARGET_FIELDS = ("probabilities", "confidence", "admitted")


def validate_fit(directory, group, steps, epochs, revision):
    result = read(directory / "result.json")
    proof = read(directory / "verification.json")
    if (proof["passed"] is not True or proof["full_test_count"] != 10000
            or proof["optimizer_steps"] != steps or result["revision"] != revision
            or result["requested_optimizer_steps"] != steps or result["training"]["epochs"] != epochs
            or result["test_accuracy_percent"] != proof["recomputed_accuracy_percent"]):
        raise ValueError(f"Full-test or training audit failed for {directory}")
    checkpoint = checked_file(Path(result["checkpoint"]), result["checkpoint_sha256"])
    with np.load(checkpoint.with_name("membership.npz"), allow_pickle=False) as saved:
        for field, value in group.items():
            np.testing.assert_array_equal(saved[field], value)
    return {
        "test_accuracy_percent": result["test_accuracy_percent"],
        "best_epoch": result["training"]["best_epoch"],
        "validation_accuracy_percent": result["training"]["best_validation_accuracy_percent"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "prediction_sha256": result["test_predictions_sha256"],
        "result_sha256": checksum(directory / "result.json"),
    }


def train(archive, targets, output, index, check_only):
    from deep_al.vast_noisy.balanced_compare20261120 import worker as balanced
    from deep_al.vast_noisy.factorial20261031 import core
    from deep_al.vast_noisy.submission20261019 import run as source

    budget, condition = CASES[index]
    name = f"b{budget}_{condition}"
    plan = read(archive / "plan.json")
    revision, epochs = plan["code"]["revision"], plan["reference_epochs"]
    case, screen = archive / "cells" / name, targets / name
    prepared = case / "inputs"
    report, _, arrays, cfg, _, labeled, unlabeled, validation, observed, val_labels = source._inputs(
        prepared, read(prepared / "preparation.json")["rounds"][0],
    )
    if (report["code"]["revision"] != revision or report["config_sha256"] != plan["config_sha256"]
            or cfg.OPTIM.MAX_EPOCH != epochs or epochs != 300 or cfg.TRAIN.BATCH_SIZE != 100
            or cfg.TRAIN.EVAL_PERIOD != 2 or cfg.RNG_SEED != 0):
        raise ValueError("Archived student configuration changed")
    original = case / f"A_{DEPTH}.npz"
    original_sha = read(case / "fixed_targets.json")["hashes"][original.name]
    screened = read(screen / "result.json")
    if screened["original_sha256"] != original_sha or screened["depth"] != DEPTH:
        raise ValueError("Candidate was derived from a different teacher")
    paths = (checked_file(original, original_sha),
             checked_file(screen / "targets.npz", screened["revised_sha256"]))
    both = []
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            both.append({field: saved[field] for field in TARGET_FIELDS})
    raw, sqrt = both
    for field in ("admitted", "confidence"):
        np.testing.assert_array_equal(raw[field], sqrt[field])
    np.testing.assert_array_equal(raw["probabilities"][labeled], sqrt["probabilities"][labeled])
    selected = np.r_[labeled, unlabeled]
    group = core.membership(raw["probabilities"], labeled, observed, unlabeled, selected,
                            raw["admitted"], gate=True, hybrid=False)
    batches = math.ceil(len(group["indices"]) / cfg.TRAIN.BATCH_SIZE)
    steps = epochs * batches
    comparison = {
        "case": name, "budget": budget, "condition": condition, "seed": cfg.RNG_SEED,
        "depth": DEPTH, "epochs": epochs, "optimizer_steps": steps,
        "n_labeled": len(group["selected_labeled"]), "n_admitted": len(group["selected_unlabeled"]),
        "revision": revision, "config_sha256": plan["config_sha256"], "test_count": 10000,
        "original_target_sha256": checksum(paths[0]), "sqrt_target_sha256": checksum(paths[1]),
        "source_sha256": checksum(Path(__file__)),
        "archived_reference": validate_fit(case / str(DEPTH) / "A", group, steps, epochs, revision),
    }
    print(json.dumps(comparison, allow_nan=False), flush=True)
    if check_only:
        return

    import torch

    comparison["gpu"] = torch.cuda.get_device_name()
    destination = output / name
    arms = {"sqrt": sqrt}
    if index == 2:
        arms["reference_refit"] = raw
    for arm, target in arms.items():
        print(f"{name} {arm}: {steps} optimizer steps", flush=True)
        fit_dir = destination / arm
        balanced.fit(fit_dir, prepared, plan, arrays, cfg, target, group, validation, val_labels,
                     steps, 2 * batches)
        comparison[arm] = validate_fit(fit_dir, group, steps, epochs, revision)
        print(json.dumps(comparison[arm], allow_nan=False), flush=True)
    summary = destination / "comparison.json"
    temporary = summary.with_suffix(".tmp")
    temporary.write_text(json.dumps(comparison, allow_nan=False, indent=2) + "\n")
    temporary.replace(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int, choices=CELLS)
    parser.add_argument("archive", type=Path)
    parser.add_argument("targets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    train(args.archive, args.targets, args.output, args.index, args.check_only)
