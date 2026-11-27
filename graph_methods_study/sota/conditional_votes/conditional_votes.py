"""Compress only graph votes from acquired annotations flagged by AUM."""

import argparse
import json
from pathlib import Path

import numpy as np

import netta_baselines as reference

METHODS = ("npc", "weighted_npc", "maxherding_nas")


def vote_components(paths, accepted, targets):
    trusted = paths[:, accepted] @ targets[accepted]
    flagged = paths[:, ~accepted]
    noisy_targets = targets[~accepted]
    return {
        "conditional": trusted + np.sqrt(flagged) @ noisy_targets,
        "half": trusted + .5 * (flagged @ noisy_targets),
        "hard": trusted,
    }


def evaluate(order, rows, walk, project, truth):
    anchors = np.searchsorted(rows, order["rows"])
    np.testing.assert_array_equal(rows[anchors], order["rows"])
    targets = np.eye(reference.CLASSES)[order["labels"]]
    matrix = np.zeros((len(rows), len(anchors)), dtype=np.float32)
    matrix[anchors, np.arange(len(anchors))] = 1
    for depth in range(1, 7):
        if depth in (3, 6):
            paths = project @ matrix
            for budget in reference.BUDGETS:
                noisy = order["masks"][str(budget)]["noisy_rows"]
                accepted = ~np.isin(order["rows"][:budget], noisy)
                values = vote_components(paths[:, :budget], accepted, targets[:budget])
                counts = {k: int((v.argmax(axis=1) == truth).sum()) for k, v in values.items()}
                old = next(v for v in order["scores"] if v["budget"] == budget and v["training_filter"] == "aum")
                if old["retained"] and old["correct"][f"raw{depth}"] != counts["hard"]:
                    raise ValueError("AUM hard-filter reference failed parity")
                yield {"budget": budget, "depth": depth, "correct": counts}
        if depth < 6:
            matrix = walk @ matrix


def run(condition, archive, output):
    import active_audit as source

    rows, nns, test, _, _, _, oracle, identity = reference.inputs(condition)
    walk, project = (reference.path_matrix(neighbors, len(rows)) for neighbors in (nns, test))
    truth = np.load(source.BANK / "test_targets.npy", allow_pickle=False)
    arms = {}
    for arm in METHODS:
        path = archive / condition / f"{arm}.json"
        original = reference.load_json(path)
        if any(original["provenance"][k] != value for k, value in identity.items()):
            raise ValueError("NPC results came from different inputs")
        np.testing.assert_array_equal(oracle[original["rows"]], original["labels"])
        arms[arm] = {"records": list(evaluate(original, rows, walk, project, truth)),
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
