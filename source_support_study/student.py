"""Compare two equal-size pseudo-label shortlists with the original VAST student."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .replay import DEPTH, checked_file, checksum, read, run_case

CASES = ((500, "sym50"), (1000, "sym50"), (1000, "sym0"), (1000, "asym50"))
ARMS = ("confidence", "source_stability")


def run(archive, output, budget, condition, epochs):
    from deep_al.vast_noisy.balanced_compare20261120 import worker as balanced
    from deep_al.vast_noisy.factorial20261031 import core, worker as audit
    from deep_al.vast_noisy.submission20261019 import run as source

    plan = read(archive / "plan.json")
    ident = f"b{budget}_{condition}"
    case = archive / "cells" / ident
    prepared = case / "inputs"
    report, _, arrays, cfg, _, labeled, unlabeled, validation, observed, val_labels = source._inputs(
        prepared, read(prepared / "preparation.json")["rounds"][0],
    )
    if (report["code"]["revision"] != plan["code"]["revision"]
            or report["config_sha256"] != plan["config_sha256"]
            or cfg.TRAIN.EVAL_PERIOD != 2 or cfg.TRAIN.BATCH_SIZE != 100):
        raise ValueError("Archived training protocol changed")
    target_path = case / f"A_{DEPTH}.npz"
    target_sha256 = read(case / "fixed_targets.json")["hashes"][target_path.name]
    with np.load(checked_file(target_path, target_sha256), allow_pickle=False) as saved:
        targets = {name: saved[name] for name in ("probabilities", "confidence", "admitted")}
    if not (output / "selection/summary.json").is_file():
        run_case(archive, output / "selection", budget, condition)
    selection_path = output / "selection/membership.npz"
    selection_summary = read(output / "selection/summary.json")
    if selection_summary["checkpoint_sha256"] != target_sha256:
        raise ValueError("Selection was derived from a different teacher")
    predicted = targets["probabilities"].argmax(axis=1)
    cfg = cfg.clone()
    cfg.OPTIM.MAX_EPOCH = epochs
    results = {}
    original_quota = None
    for arm in ARMS:
        with np.load(selection_path, allow_pickle=False) as members:
            admitted = members[f"top10_{arm}"]
        result = selection_summary["top10"][arm]
        if (len(admitted) != result["admitted"] or len(admitted) != math.ceil(.10 * len(unlabeled))
                or not np.isin(admitted, unlabeled).all()):
            raise ValueError("Wrong shortlist size or partition")
        quota = np.bincount(predicted[admitted], minlength=100)
        if original_quota is None:
            original_quota = quota
        np.testing.assert_array_equal(quota, original_quota)
        if arm == "confidence" and result["replaced"]:
            raise ValueError("Confidence shortlist differs from its reference")
        mask = np.zeros(len(predicted), dtype=bool)
        mask[admitted] = True
        group = core.membership(targets["probabilities"], labeled, observed, unlabeled,
                                np.r_[labeled, unlabeled], mask, gate=True, hybrid=False)
        batches = math.ceil(len(group["indices"]) / cfg.TRAIN.BATCH_SIZE)
        steps = epochs * batches
        print(json.dumps({"case": ident, "arm": arm, "steps": steps, "status": "training"}), flush=True)
        destination = output / arm
        fit = balanced.fit(destination, prepared, plan, arrays, cfg, targets, group,
                           validation, val_labels, steps, 2 * batches)
        proof = read(destination / "verification.json")
        if (not proof["passed"] or proof["full_test_count"] != 10000
                or proof["optimizer_steps"] != steps):
            raise ValueError("Student fit failed its audit")
        checkpoint = Path(fit["checkpoint"])
        with np.load(checkpoint.with_name("membership.npz"), allow_pickle=False) as fitted:
            for field, value in group.items():
                np.testing.assert_array_equal(fitted[field], value)
        results[arm] = {
            "test_accuracy_percent": fit["test_accuracy_percent"],
            "best_epoch": fit["training"]["best_epoch"],
            "optimizer_steps": steps, "labeled": len(labeled), "unlabeled": len(admitted),
            "checkpoint_sha256": fit["checkpoint_sha256"], "pseudo_precision": result["precision"],
        }
        print(json.dumps({"case": ident, "arm": arm, "status": "verified", **results[arm]}), flush=True)
    audit.write(output / "comparison.json", {
        "case": ident, "budget": budget, "condition": condition, "epochs": epochs,
        "target_sha256": target_sha256, "selection_sha256": checksum(selection_path), "arms": results,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True, choices=range(len(CASES)))
    parser.add_argument("--epochs", type=int, default=300, choices=(2, 300))
    args = parser.parse_args()
    budget, condition = CASES[args.index]
    run(args.archive, args.output / f"b{budget}_{condition}", budget, condition, args.epochs)


if __name__ == "__main__":
    main()
