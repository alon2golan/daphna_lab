"""Screen pseudo-label admission on the completed CIFAR-100 fixed-depth study."""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sparse

from .stability import rerank, source_stability

BUDGETS = (200, 500, 1000, 2000)
CONDITIONS = ("sym0", "sym20", "sym50", "asym20", "asym50")
CELLS = tuple(product(BUDGETS, CONDITIONS))
DEPTH = 5


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def checked_file(path, expected):
    if checksum(path) != expected:
        raise ValueError(f"Reference artifact changed: {path}")
    return path


def teacher(graph, initial, labeled, observed):
    from deep_al.vast_noisy.submission20261019 import propagation

    masks = []

    def record(step, probabilities, senders, receivers, frozen, newly_frozen):
        if step:
            masks.append((senders, receivers))

    evidence, _, _ = propagation.run(
        graph, initial, labeled, observed, rounds=DEPTH, early_stop=False,
        mutable_labeled=False, observer=record,
    )
    return evidence, masks


def run_case(archive, output, budget, condition):
    plan = read(archive / "plan.json")
    case = archive / "cells" / f"b{budget}_{condition}"
    if not (case / "verified.json").is_file():
        raise ValueError(f"Reference cell is incomplete: {case.name}")
    prepared = case / "inputs"
    provenance = read(prepared / "provenance.json")
    files = read(prepared / "preparation.json")["prepared_sha256"]

    def load_input(name):
        return np.load(checked_file(prepared / name, files[name]), allow_pickle=False)

    labeled = load_input("partitions/labeled_indices.npy")
    unlabeled = load_input("partitions/unlabeled_indices.npy")
    validation = load_input("partitions/validation_indices.npy")
    observed = load_input("observed_labeled_labels.npy")
    initial_path = prepared / "initial_evidence.npy"
    initial = np.load(initial_path, allow_pickle=False)
    if (len(labeled) != budget or len(validation) != 5000 or len(observed) != budget
            or not np.array_equal(np.sort(np.r_[labeled, unlabeled, validation]), np.arange(50000))):
        raise ValueError("Unexpected acquisition, unlabeled and validation partition")

    graph_info = plan["graph"]
    graph = sparse.load_npz(checked_file(Path(graph_info["path"]), graph_info["kernel_sha256"])).tocsr()
    name = f"A_{DEPTH}.npz"
    checkpoint = checked_file(case / name, read(case / "fixed_targets.json")["hashes"][name])
    with np.load(checkpoint, allow_pickle=False) as saved:
        probabilities, admitted = saved["probabilities"], saved["admitted"]

    evidence, masks = teacher(graph, initial, labeled, observed)
    rebuilt = evidence / evidence.sum(axis=1, keepdims=True)
    parity_error = float(np.max(np.abs(rebuilt - probabilities)))
    if not np.isfinite(parity_error) or parity_error > 3e-6 or len(masks) != DEPTH:
        raise ValueError(f"Could not recover the recorded teacher masks: {parity_error:g}")

    margin, fraction, winning_share, reconstruction_error = source_stability(
        graph, labeled, observed, probabilities, masks,
    )
    classes = probabilities.shape[1]
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    plain_margin = confidence - np.partition(probabilities, -2, axis=1)[:, -2]
    candidates = np.zeros(len(initial), dtype=bool)
    candidates[unlabeled] = True
    candidates &= confidence > 1.0 / classes + 1e-6
    selections = {
        "itai": admitted,
        "confidence": rerank(candidates, admitted, predicted, confidence, confidence, classes),
        "margin": rerank(candidates, admitted, predicted, plain_margin, confidence, classes),
        "source_stability": rerank(candidates, admitted, predicted, margin, confidence, classes),
        "bounded_discount": rerank(candidates, admitted, predicted,
                                   confidence * (1.0 - 0.25 * winning_share), confidence, classes),
    }
    quotas = np.bincount(predicted[admitted], minlength=classes)

    # Truth first enters here, after the admission decisions are complete.
    truth = np.load(checked_file(Path(provenance["files"]["train.labels"]),
                                 plan["features"]["train.labels"]), allow_pickle=False)
    correct = predicted == truth
    result = {
        "case": case.name, "budget": budget, "condition": condition, "depth": DEPTH,
        "checkpoint_sha256": checksum(checkpoint), "initial_sha256": checksum(initial_path),
        "graph_sha256": graph_info["kernel_sha256"], "teacher_max_abs_error": parity_error,
        "source_max_abs_error": reconstruction_error, "pool_count": len(unlabeled),
        "seed_accuracy": float(np.mean(observed == truth[labeled])),
        "supported_classes": int(np.count_nonzero(quotas)),
        "arms": {},
    }
    for arm, selection in selections.items():
        np.testing.assert_equal(quotas, np.bincount(predicted[selection], minlength=classes))
        hits = np.bincount(predicted[selection & correct], minlength=classes)
        result["arms"][arm] = {
            "admitted": int(selection.sum()),
            "precision": float(correct[selection].mean()),
            "macro_precision": float(np.mean(hits[quotas > 0] / quotas[quotas > 0])),
            "replaced": int(np.sum(selection & ~admitted)),
            "mean_anchor_fraction": float(fraction[selection].mean()),
            "stable_fraction": float((margin[selection] > 0).mean()),
        }

    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(f"Completed output already exists: {output}")
    with (output / "membership.npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, **{arm: np.flatnonzero(mask) for arm, mask in selections.items()})
    (output / "membership.npz.tmp").replace(output / "membership.npz")
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "summary.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    case = parser.add_mutually_exclusive_group(required=True)
    case.add_argument("--index", type=int, choices=range(len(CELLS)))
    case.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    selected = ((200, "sym50"), (1000, "sym50")) if args.pilot else (CELLS[args.index],)
    for budget, condition in selected:
        output = args.output / f"b{budget}_{condition}"
        result = run_case(args.archive, output, budget, condition)
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
