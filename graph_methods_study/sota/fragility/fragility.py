"""Replace the target weight in exact Gaussian herding with source fragility."""

import argparse
import json
from pathlib import Path

import numpy as np

CONDITIONS = ("clean", "sym20", "sym30", "sym40", "sym50",
              "asym20", "asym30", "asym40", "asym50", "cifar100n")
BUDGETS = (400, 800, 1200, 1600, 2000)
BATCH = 100


def source_fragility(paths, labels, classes):
    """Return one plus the probability of losing a strict lead on deletion."""
    row = np.repeat(np.arange(paths.shape[0]), np.diff(paths.indptr))
    cls = labels[paths.indices]
    weight = paths.data
    votes = np.zeros((paths.shape[0], classes))
    np.add.at(votes, (row, cls), weight)
    winner = votes.argmax(axis=1)
    largest_two = np.partition(votes, classes - 2, axis=1)[:, -2:]
    gap = largest_two[:, 1] - largest_two[:, 0]
    can_break = (winner[row] == cls) & (weight >= gap[row])
    broken = np.bincount(row[can_break], weights=weight[can_break], minlength=len(votes))
    total = votes.sum(axis=1)
    return 1 + np.divide(broken, total, out=np.zeros_like(total), where=total > 0)


def run(condition, archive, geometry, exact_orders, human, output):
    from scipy.sparse import csr_matrix
    from deep_al.vast_noisy.js_ablation20261102.geometry import dense_select

    import active_audit as source
    from evaluate import GEOMETRY_SHA256, HUMAN_SHA256, RBF_SHA256, evaluate_order, walk_matrix

    if source.sha256(geometry) != GEOMETRY_SHA256:
        raise ValueError("Frozen 13NN geometry changed")
    with np.load(geometry, allow_pickle=False) as saved:
        rows, neighbors, heldout = (saved[k] for k in ("rows", "neighbors", "heldout_neighbors"))
    plan = json.loads((archive / "plan.json").read_text())
    gaussian = Path(plan["geometry"])
    metadata = gaussian / "geometry.json"
    if source.sha256(metadata) != plan["geometry_ready"]:
        raise ValueError("Frozen Gaussian manifest changed")
    matrix_artifact = json.loads(metadata.read_text())["files"]["rbf"]
    if matrix_artifact["sha256"] != RBF_SHA256:
        raise ValueError("Wrong complete Gaussian affinity")
    np.testing.assert_array_equal(np.load(archive / "inputs/eligible.npy", allow_pickle=False), rows)
    exact = json.loads(exact_orders.read_text())
    if exact["geometry_sha256"] != RBF_SHA256 or exact["common_prefix"] != BATCH:
        raise ValueError("Exact order came from a different Gaussian geometry")
    warm = np.searchsorted(rows, exact["maxherding"][:BATCH])
    np.testing.assert_array_equal(rows[warm], exact["maxherding"][:BATCH])
    rbf = np.load(gaussian / matrix_artifact["file"], mmap_mode="r", allow_pickle=False)

    canonical = np.load(source.BANK / "train_targets.npy", allow_pickle=False)
    truth = np.load(source.BANK / "test_targets.npy", allow_pickle=False)
    if condition in plan["noises"]:
        oracle = plan["noises"][condition]
        label_hash = oracle["observed_sha256"]
        path = Path(oracle["observed_path"])
        if source.sha256(path) != label_hash:
            raise ValueError("Archived annotations changed")
        observed = np.load(path, allow_pickle=False)
    elif condition == "cifar100n":
        label_hash = source.sha256(human)
        if label_hash != HUMAN_SHA256:
            raise ValueError("Frozen human annotations changed")
        observed = np.load(human, allow_pickle=False)
    else:
        groups = list(json.loads(source.GROUPS.read_text()).values())
        successor = np.empty(source.CLASSES, dtype=np.int64)
        for group in groups:
            successor[group] = np.roll(group, -1)
        observed = source.corrupt_labels(canonical, condition, .4, 0, successor)
        label_hash = None

    walk = walk_matrix(neighbors, len(rows))
    order = warm
    rounds = []
    while len(order) < max(BUDGETS):
        n = len(order)
        starts = csr_matrix((np.ones(n), (order, np.arange(n))), shape=(len(rows), n))
        paths = walk @ (walk @ (walk @ starts))
        weights = source_fragility(paths, observed[rows[order]], source.CLASSES)
        picked, gains = dense_select(rbf, order, None, BATCH, weights)
        order = np.r_[order, picked]
        rounds.append({"budget": len(order), "mean_weight": float(weights.mean()),
                       "greedy_gain": float(gains.sum())})
        print(condition, len(order), round(weights.mean(), 3), flush=True)
    if len(np.unique(order)) != max(BUDGETS):
        raise ValueError("Duplicate acquisition")
    test = source.unit_rows(np.load(source.BANK / "test_features.npy", allow_pickle=False))
    train = np.load(source.BANK / "train_features.npy", mmap_mode="r", allow_pickle=False)
    predict = walk_matrix(heldout, len(rows))
    arms = {}
    for method, selected_rows in (("fragility", rows[order]), ("maxherding", exact["maxherding"]),
                                  ("backup_coverage", exact["backup_coverage"])):
        aligned = source.replay_order(neighbors, rows, max(BUDGETS), 0, selected_rows)
        arms[method] = evaluate_order(train, test, {condition: observed}, canonical, truth, BUDGETS,
                                      aligned, rows, walk, predict, method)
    packet = {"condition": condition, "seed": 0, "test_count": len(truth), "source_sha256": source.sha256(Path(__file__)),
              "gaussian_sha256": RBF_SHA256, "geometry_sha256": GEOMETRY_SHA256, "labels_sha256": label_hash,
              "archive_order_sha256": source.sha256(exact_orders), "acquisition_rows": rows[order].tolist(),
              "arms": arms, "rounds": rounds}
    temporary = output.with_suffix(".tmp")
    with temporary.open("x") as stream:
        json.dump(packet, stream, allow_nan=False)
    temporary.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int, choices=range(len(CONDITIONS)))
    parser.add_argument("archive", type=Path)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("exact_orders", type=Path)
    parser.add_argument("human", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(CONDITIONS[args.index], args.archive, args.geometry, args.exact_orders, args.human, args.output)
