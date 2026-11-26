"""Vary wrong-label concentration while freezing corrupted rows and geometry."""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sparse

from .replay import checked_file, read, summarize, teacher
from .stability import rerank, source_stability

BUDGET = 800
SEEDS = tuple(range(5))


def noise_channels(truth, groups, seed):
    rng = np.random.default_rng(35_000 + seed)
    classes = sum(map(len, groups))
    corrupt = rng.choice(len(truth), len(truth) // 2, replace=False)
    draws = rng.integers(classes - 1, size=len(corrupt))
    near = np.empty(classes, dtype=int)
    for group in groups:
        near[group] = np.roll(group, -1)
    permutation = rng.permutation(classes)
    scattered = np.empty(classes, dtype=int)
    scattered[permutation] = np.roll(permutation, -1)
    mapping = {
        "uniform": draws + (draws >= truth[corrupt]),
        "scattered_cycle": scattered[truth[corrupt]],
        "superclass_cycle": near[truth[corrupt]],
    }
    for name, replacements in mapping.items():
        if not np.all(replacements != truth[corrupt]):
            raise ValueError("A corruption left its label unchanged")
        observed = truth.copy()
        observed[corrupt] = replacements
        yield name, observed


def score_channel(graph, labeled, unlabeled, labels, truth):
    from deep_al.vast_noisy.submission20261019 import adapters

    initial = adapters.seed_evidence(graph, labeled, labels, 100)
    evidence, masks = teacher(graph, initial, labeled, labels)
    q = evidence / evidence.sum(axis=1, keepdims=True)
    margin, fraction, _, error = source_stability(graph, labeled, labels, q, masks)
    confidence = q.max(axis=1)
    prediction = q.argmax(axis=1)
    candidate = np.zeros(len(q), dtype=bool)
    candidate[unlabeled] = confidence[unlabeled] > .01 + 1e-6
    count = len(unlabeled) // 10
    rows = np.flatnonzero(candidate)
    if len(rows) < count:
        raise ValueError("Too few informed predictions at fixed coverage")
    top = rows[np.lexsort((rows, -confidence[rows]))[:count]]
    reference = np.zeros_like(candidate)
    reference[top] = True
    raw_margin = confidence - np.partition(q, -2, axis=1)[:, -2]
    arms = {
        "reference": reference,
        "margin": rerank(candidate, reference, prediction, raw_margin, confidence, 100),
        "source_stability": rerank(candidate, reference, prediction, margin, confidence, 100),
    }
    return {
        "seed_accuracy": float((labels == truth[labeled]).mean()), "error": error,
        "arms": summarize(arms, prediction, prediction == truth, fraction, margin, 100),
    }


def run(projects, audit, output, seed):
    archive = projects / "cifar100_balanced_depths_20261123/study"
    plan = read(archive / "plan.json")
    provenance = read(archive / "cells/b1000_sym50/inputs/provenance.json")
    truth = np.load(checked_file(Path(provenance["files"]["train.labels"]),
                                 plan["features"]["train.labels"]), allow_pickle=False)
    geometry = plan["graph"]
    graph = sparse.load_npz(checked_file(Path(geometry["path"]), geometry["kernel_sha256"])).tocsr()
    eligible = np.load(projects / "cifar100_herding_20261119/study/inputs/eligible.npy", allow_pickle=False)
    audit_file = audit / f"seed_{seed}.json"
    labeled = np.asarray(read(audit_file)["acquisition_rows"][:BUDGET])
    if len(eligible) != 45000 or len(np.unique(labeled)) != BUDGET or not np.isin(labeled, eligible).all():
        raise ValueError("Acquisition is not an aligned seed from the audit replay")
    unlabeled = np.setdiff1d(eligible, labeled)
    groups = list(read(projects / "vast_noisy_inputs_fc280c3/cifar100_superclasses.json").values())
    if len(groups) != 20 or any(len(group) != 5 for group in groups) or sorted(sum(groups, [])) != list(range(100)):
        raise ValueError("Invalid CIFAR-100 superclass mapping")

    results = []
    for channel, observed in noise_channels(truth, groups, seed):
        stats = score_channel(graph, labeled, unlabeled, observed[labeled], truth)
        result = {"seed": seed, "budget": BUDGET, "channel": channel, **stats}
        results.append(result)
        print(json.dumps(result, allow_nan=False), flush=True)
    with output.open("x") as stream:
        json.dump({"seed": seed, "graph_sha256": geometry["kernel_sha256"], "rows": results}, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    args = parser.parse_args()
    run(args.projects, args.audit, args.output, args.seed)


if __name__ == "__main__":
    main()
