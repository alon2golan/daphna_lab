"""Add a second predicted-clean covering center to the fixed-radius NPC objective."""

import argparse
from itertools import product
import hashlib
import json
from pathlib import Path

import numpy as np

import netta_baselines as reference

FRESH_CONDITIONS = ("sym30", "sym40", "sym50", "asym30", "asym40", "asym50")
CASES = tuple((0, condition) for condition in reference.CONDITIONS) + tuple(
    product((51, 52, 53), FRESH_CONDITIONS))
VARIANTS = {"npc_backup": False, "weighted_backup": True}
BATCH = reference.BATCH
BUDGETS = reference.BUDGETS


def radius_backup_batch(graph, incoming, labeled, noisy, weighted, rng, count=BATCH, backup=.5):
    """Value first and second distinct coverage with the original NPC demand factors."""
    demand = np.ones(graph.shape[0])
    if weighted and noisy.any():
        demand[graph[labeled[noisy]].indices] = 1 - noisy.mean()
    coverage = np.asarray(graph[labeled[~noisy]].sum(axis=0)).ravel().astype(np.int32)
    scale = np.array([1., backup, 0.])
    gain = graph @ (demand * scale[np.minimum(coverage, 2)])
    gain[labeled] = -np.inf
    chosen = []
    for _ in range(count):
        ties = np.flatnonzero(np.isclose(gain, gain.max(), rtol=0, atol=1e-10))
        row = int(rng.choice(ties))
        chosen.append(row)
        gain[row] = -np.inf
        targets = graph.indices[graph.indptr[row]:graph.indptr[row + 1]]
        useful = targets[coverage[targets] < 2]
        for col in useful:
            left, right = incoming.indptr[col:col + 2]
            delta = demand[col] * (scale[coverage[col] + 1] - scale[coverage[col]])
            gain[incoming.indices[left:right]] += delta
        coverage[useful] += 1
    return np.asarray(chosen)


def backup_order(features, population, pool, graph, baseline, detector, weighted):
    order = baseline[:BATCH].copy()
    known = np.full(len(features), -1, dtype=int)
    known[order] = pool[population[order]]
    incoming = graph.tocsc()
    rng = np.random.RandomState(0)
    masks = {}
    while True:
        budget = len(order)
        remaining = np.setdiff1d(np.arange(len(features)), order)
        noisy, info = detector(features, known, order, remaining, num_classes=reference.CLASSES,
                               seed=0, device="cpu")
        if len(noisy) != budget or not np.isfinite(info.get("threshold", 0)):
            raise ValueError("Invalid AUM noise mask")
        if budget in BUDGETS:
            masks[budget] = {"noisy_rows": population[order[noisy]].tolist()}
        if budget == max(BUDGETS):
            break
        query = radius_backup_batch(graph, incoming, order, noisy, weighted, rng)
        if len(np.unique(query)) != BATCH or not np.isin(query, remaining).all():
            raise ValueError("Invalid backup query")
        known[query] = pool[population[query]]
        order = np.r_[order, query]
    return {"rows": population[order].tolist(), "labels": known[order].tolist(), "masks": masks}


def common_scores(artifact, population, walk, heldout, features, test, truth):
    from active_audit import unit_rows

    positions = np.searchsorted(population, artifact["rows"])
    np.testing.assert_array_equal(population[positions], artifact["rows"])
    labels = np.asarray(artifact["labels"])
    basis = np.zeros((len(population), len(positions)), dtype=np.float32)
    basis[positions, np.arange(len(positions))] = 1
    votes = {}
    for depth in range(1, 7):
        if depth in (3, 6):
            paths = heldout @ basis
            votes[f"raw{depth}"] = paths
            votes[f"sqrt{depth}"] = np.sqrt(paths.astype(np.float64))
        if depth < 6:
            basis = walk @ basis
    similarities = test @ unit_rows(features[positions]).T
    eye = np.eye(reference.CLASSES)
    for budget in BUDGETS:
        noisy = artifact["masks"][str(budget) if str(budget) in artifact["masks"] else budget]["noisy_rows"]
        clean = ~np.isin(artifact["rows"][:budget], noisy)
        for filt, keep in (("all", np.ones(budget, dtype=bool)), ("aum", clean)):
            scores = {}
            if keep.any():
                indicator = eye[labels[:budget][keep]]
                for name, paths in votes.items():
                    prediction = (paths[:, :budget][:, keep] @ indicator).argmax(axis=1)
                    scores[name] = int((prediction == truth).sum())
                nearest = similarities[:, :budget][:, keep].argmax(axis=1)
                scores["1nn"] = int((labels[:budget][keep][nearest] == truth).sum())
            yield {"budget": budget, "training_filter": filt, "retained": int(keep.sum()), "correct": scores}


def run(seed, condition, archived, output):
    import active_audit as graph_source
    from deep_al.pycls.al import lowbudget_aum
    import torch

    torch.set_num_threads(2)
    detector = lowbudget_aum.lowbudget_aum_noisy_mask
    reference.checked(Path(lowbudget_aum.__file__), reference.DETECTOR_SHA256)
    base = condition if seed == 0 else "clean"
    population, nns, test_nns, geometry, rbf, baseline, pool, provenance = reference.inputs(base)
    canonical = np.load(graph_source.BANK / "train_targets.npy", allow_pickle=False)
    truth = np.load(graph_source.BANK / "test_targets.npy", allow_pickle=False)
    if seed:
        groups = list(json.loads(graph_source.GROUPS.read_text()).values())
        successor = np.empty(reference.CLASSES, dtype=np.int64)
        for group in groups:
            successor[group] = np.roll(group, -1)
        pool = graph_source.corrupt_labels(canonical, condition, int(condition[-2:]) / 100, seed, successor)
        provenance["observed_sha256"] = hashlib.sha256(np.asarray(pool, dtype="<i8").tobytes()).hexdigest()
        provenance["observed_hash_format"] = "int64_le_array"
    features = np.load(graph_source.BANK / "train_features.npy", mmap_mode="r", allow_pickle=False)[population]
    test = graph_source.unit_rows(np.load(graph_source.BANK / "test_features.npy", allow_pickle=False))
    walk = reference.path_matrix(nns, len(population))
    heldout = reference.path_matrix(test_nns, len(population))
    methods = (*reference.METHODS, *VARIANTS) if not seed else ("npc", "weighted_npc", *VARIANTS)
    results = {}
    for method in methods:
        archived_fit = None
        if method in VARIANTS:
            order = backup_order(features, population, pool, geometry, baseline, detector, VARIANTS[method])
        elif seed:
            order = reference.trajectory(method, features, population, pool, geometry, rbf, baseline, detector)
        else:
            archived_fit = archived / condition / f"{method}.json"
            order = reference.load_json(archived_fit)
            original = order["provenance"]
            if any(original[k] != value for k, value in provenance.items()):
                raise ValueError("NPC baseline has different inputs")
        if len(set(order["rows"])) != 2000 or order["rows"][:BATCH] != population[baseline[:BATCH]].tolist():
            raise ValueError("Invalid common acquisition budget")
        np.testing.assert_array_equal(pool[order["rows"]], order["labels"])
        scores = list(common_scores(order, population, walk, heldout, features, test, truth))
        if archived_fit:
            for actual, expected in zip(scores, order["scores"], strict=True):
                for key in ("training_filter", "retained", "budget"):
                    if actual[key] != expected[key]:
                        raise ValueError("NPC score uses different masks")
                if any(actual["correct"][head] != value for head, value in expected["correct"].items()):
                    raise ValueError("Original Netta classifier score failed parity")
        results[method] = {"order_sha256": hashlib.sha256(np.asarray(order["rows"], dtype="<i8").tobytes()).hexdigest(),
                           "scores": scores, "observed_wrong": {
                               b: int(np.count_nonzero(np.asarray(order["labels"][:b]) != canonical[order["rows"][:b]]))
                               for b in BUDGETS}}
        print(seed, condition, method, "verified", flush=True)
    artifact = dict(provenance, seed=seed, condition=condition, test_count=len(truth),
                    source_sha256=reference.sha256(__file__), netta_sha256=reference.sha256(reference.__file__),
                    detector_sha256=reference.DETECTOR_SHA256, arms=results)
    with output.open("x") as out:
        json.dump(artifact, out, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int, choices=range(len(CASES)))
    parser.add_argument("archived", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(*CASES[args.index], args.archived, args.output)
