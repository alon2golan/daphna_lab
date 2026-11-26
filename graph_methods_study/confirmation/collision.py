"""Replay graph votes with the corrupted rows fixed across three noise channels."""

import argparse
import json
from pathlib import Path

import numpy as np

import active_audit as source
from evaluate import GEOMETRY_SHA256, evaluate_order, walk_matrix
from source_support_study import channel

SEEDS = range(40, 50)
BUDGETS = (800, 1200, 1600)


def replay(seed, geometry, saved_order, output):
    if source.sha256(geometry) != GEOMETRY_SHA256:
        raise ValueError("Archived geometry changed")
    with np.load(geometry, allow_pickle=False) as saved:
        rows, neighbors, heldout = (saved[name] for name in ("rows", "neighbors", "heldout_neighbors"))
    reference = json.loads(saved_order.read_text())
    acquired = reference["acquisition_rows"]["seeded_coverage"]
    if (reference["seed"] != seed or reference["geometry_sha256"] != GEOMETRY_SHA256
            or len(acquired) != max(BUDGETS)):
        raise ValueError("Unexpected confirmation acquisition")
    order = source.replay_order(neighbors, rows, max(BUDGETS), seed, acquired)
    canonical = np.load(source.BANK / "train_targets.npy", allow_pickle=False)
    truth = np.load(source.BANK / "test_targets.npy", allow_pickle=False)
    groups = list(json.loads(source.GROUPS.read_text()).values())
    labels = dict(channel.noise_channels(canonical, groups, seed))
    wrong = [observed != canonical for observed in labels.values()]
    np.testing.assert_array_equal(wrong[0], wrong[1])
    np.testing.assert_array_equal(wrong[0], wrong[2])
    if len(truth) != 10000 or wrong[0].sum() * 2 != len(canonical):
        raise ValueError("Unexpected evaluation split or corruption count")
    test = source.unit_rows(np.load(source.BANK / "test_features.npy", allow_pickle=False))
    train = np.load(source.BANK / "train_features.npy", mmap_mode="r", allow_pickle=False)
    walk, predict = walk_matrix(neighbors, len(rows)), walk_matrix(heldout, len(rows))
    records = evaluate_order(train, test, labels, canonical, truth, BUDGETS, order, rows,
                             walk, predict, "seeded_coverage")
    artifact = {
        "seed": seed, "test_count": len(truth), "geometry_sha256": GEOMETRY_SHA256,
        "reference_sha256": source.sha256(saved_order), "source_sha256": source.sha256(Path(__file__)),
        "channel_sha256": source.sha256(Path(channel.__file__)), "rows": records,
    }
    with output.open("x") as f:
        json.dump(artifact, f, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=SEEDS)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("saved_order", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    replay(args.seed, args.geometry, args.saved_order, args.output)
