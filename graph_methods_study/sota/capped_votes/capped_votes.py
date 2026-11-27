"""Cap the strongest graph annotation at the strength of the second source."""

import argparse
import json
from pathlib import Path

import numpy as np

import netta_baselines as reference

METHODS = ("npc", "weighted_npc", "maxherding_nas")


def second_source_cap(paths):
    if paths.shape[1] < 2:
        return paths
    second = np.partition(paths, paths.shape[1] - 2, axis=1)[:, -2]
    cap = np.where(second > 0, second, np.inf)
    return np.minimum(paths, cap[:, None])


def evaluate(order, rows, walk, project, truth):
    anchors = np.searchsorted(rows, order["rows"])
    np.testing.assert_array_equal(rows[anchors], order["rows"])
    targets = np.eye(reference.CLASSES)[order["labels"]]
    basis = np.zeros((len(rows), len(anchors)), dtype=np.float32)
    basis[anchors, np.arange(len(anchors))] = 1
    for depth in range(1, 7):
        if depth in (3, 6):
            paths = project @ basis
            for budget in reference.BUDGETS:
                noisy = order["masks"][str(budget)]["noisy_rows"]
                accepted = ~np.isin(order["rows"][:budget], noisy)
                both = [("all", np.ones(budget, dtype=bool)), ("aum", accepted)]
                for label_filter, keep in both:
                    if not keep.any():
                        continue
                    weights = second_source_cap(paths[:, :budget][:, keep])
                    votes = weights @ targets[:budget][keep]
                    correct = int((votes.argmax(axis=1) == truth).sum())
                    yield {"budget": budget, "depth": depth, "training_filter": label_filter,
                           "correct": correct}
        if depth < 6:
            basis = walk @ basis


def run(condition, archive, output):
    import active_audit as source

    rows, nns, test, _, _, _, oracle, identity = reference.inputs(condition)
    walk, project = (reference.path_matrix(neighbors, len(rows)) for neighbors in (nns, test))
    truth = np.load(source.BANK / "test_targets.npy", allow_pickle=False)
    arms = {}
    for method in METHODS:
        path = archive / condition / f"{method}.json"
        order = reference.load_json(path)
        if any(order["provenance"][k] != value for k, value in identity.items()):
            raise ValueError("NPC inputs changed before vote capping")
        np.testing.assert_array_equal(oracle[order["rows"]], order["labels"])
        arms[method] = {"records": list(evaluate(order, rows, walk, project, truth)),
                        "order_artifact_sha256": reference.sha256(path)}
    artifact = dict(identity, seed=0, condition=condition, test_count=len(truth),
                    source_sha256=reference.sha256(__file__), arms=arms)
    with output.open("x") as stream:
        json.dump(artifact, stream, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int, choices=range(len(reference.CONDITIONS)))
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(reference.CONDITIONS[args.index], args.archive, args.output)
