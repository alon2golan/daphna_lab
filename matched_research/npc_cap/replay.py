"""Compare the original six-hop graph and cap on official NPC labels."""

import argparse
import json
from pathlib import Path

import numpy as np

from active_audit import unit_rows
from capped_votes import second_source_cap
from netta_baselines import path_matrix
from run_official import BANK_HASHES, BUDGETS, CONDITIONS, SEEDS, UPSTREAM_COMMIT, file_sha256

CASES = tuple((condition, seed) for seed in SEEDS for condition in CONDITIONS)
DEGREE = 13


def read(path):
    return json.loads(path.read_text())


def build_graph(inputs, destination):
    import faiss

    for name, expected in BANK_HASHES.items():
        if file_sha256(inputs / f"{name}.npy") != expected:
            raise ValueError(f"Frozen BYOL bank changed for {name}")
    train = unit_rows(np.load(inputs / "train_features.npy", allow_pickle=False))
    test = unit_rows(np.load(inputs / "test_features.npy", allow_pickle=False))
    if train.shape != (50000, 512) or test.shape != (10000, 512):
        raise ValueError("Expected full official CIFAR-100 pools")
    index = faiss.IndexFlatIP(train.shape[1])
    index.add(train)
    _, neighbors = index.search(train, DEGREE + 1)
    rows = np.arange(len(train))[:, None]
    positions = np.argsort(neighbors == rows, axis=1, kind="stable")[:, :DEGREE]
    neighbors = np.take_along_axis(neighbors, positions, axis=1)
    _, heldout = index.search(test, DEGREE)
    if (np.any(neighbors == rows) or np.any(neighbors < 0)
            or np.any(heldout < 0)):
        raise ValueError("Invalid cosine neighbor list")
    with destination.open("xb") as stream:
        np.savez_compressed(stream, neighbors=neighbors.astype(np.int32),
                            heldout=heldout.astype(np.int32), faiss_version=faiss.__version__,
                            bank=json.dumps(BANK_HASHES, sort_keys=True))
    print(f"Prepared {len(train)} training and {len(test)} test rows", flush=True)


def six_edge_counts(walk, project, anchors, block_size=128):
    """One test-to-train edge followed by five train-to-train edges."""
    result = np.empty((project.shape[0], len(anchors)), dtype=np.float32)
    for start in range(0, len(anchors), block_size):
        block = anchors[start:start + block_size]
        basis = np.zeros((walk.shape[0], len(block)), dtype=np.float32)
        basis[block, np.arange(len(block))] = 1
        for _ in range(5):
            basis = walk @ basis
        result[:, start:start + len(block)] = project @ basis
    return result


def graph_correct(paths, labels, truth):
    classes = np.eye(100, dtype=np.float32)[labels]
    raw = (paths @ classes).argmax(axis=1)
    capped = (second_source_cap(paths) @ classes).argmax(axis=1)
    return int((raw == truth).sum()), int((capped == truth).sum())


def score(study, graph_path, condition, seed, output):
    case = f"{condition}_s{seed}"
    run = study / "runs" / case
    provenance = read(run / "provenance.json")
    summary = read(run / "summary.json")
    if (provenance["upstream_commit"] != UPSTREAM_COMMIT
            or provenance["input_manifest"]["bank_sha256"] != BANK_HASHES
            or (summary["condition"], summary["seed"], summary["smoke"]) != (condition, seed, False)):
        raise ValueError(f"Unrecognized official reference in {run}")
    if tuple(record["budget"] for record in summary["cells"]) != BUDGETS:
        raise ValueError(f"Missing official budgets in {run}")
    with np.load(graph_path, allow_pickle=False) as graph:
        neighbors, heldout = graph["neighbors"], graph["heldout"]
        if json.loads(graph["bank"].item()) != BANK_HASHES:
            raise ValueError("Neighbor features changed")
    if (neighbors.shape, heldout.shape) != ((50000, DEGREE), (10000, DEGREE)):
        raise ValueError("Expected full official graph geometry")
    test_path = study / "inputs" / "test_targets.npy"
    if file_sha256(test_path) != BANK_HASHES["test_targets"]:
        raise ValueError("Held-out labels changed")
    truth = np.load(test_path, allow_pickle=False)
    with np.load(run / "fits" / "02000" / "fit_inputs.npz", allow_pickle=False) as archive:
        anchors = archive["acquired_indices"]
    paths = six_edge_counts(path_matrix(neighbors, len(neighbors)),
                            path_matrix(heldout, len(neighbors)), anchors)
    records = []
    for baseline in summary["cells"]:
        budget = baseline["budget"]
        snapshot = run / "fits" / f"{budget:05}" / "fit_inputs.npz"
        identity = read(study / "peer" / case / f"{budget:05}" / "identity_result.json")
        if (identity["fitted_model_sha256"] != baseline["fitted_model_sha256"]
                or identity["prediction_sha256"] != baseline["prediction_sha256"]
                or identity["fit_inputs_sha256"] != file_sha256(snapshot)):
            raise ValueError(f"Reference fit and acquisition differ at {case} budget {budget}")
        with np.load(snapshot, allow_pickle=False) as archive:
            order = archive["acquired_indices"]
            keep = archive["base_keep"]
            annotation = archive["observed_labels"]
        if (not np.array_equal(order, anchors[:budget]) or keep.dtype != np.bool_
                or int(keep.sum()) != baseline["kept"] or not keep.any()):
            raise ValueError(f"Acquisition or official AUM keep mask differs at {case} budget {budget}")
        raw, capped = graph_correct(paths[:, :budget][:, keep], annotation[keep], truth)
        records.append(dict(budget=budget, retained=int(keep.sum()),
                            official=int(round(baseline["accuracy"] * len(truth) / 100)),
                            raw=raw, cap=capped, input_sha256=identity["fit_inputs_sha256"],
                            official_prediction_sha256=baseline["prediction_sha256"]))
    result = dict(condition=condition, seed=seed, test_count=len(truth), records=records,
                  graph_sha256=file_sha256(graph_path), source_sha256=file_sha256(Path(__file__)),
                  cap_sha256=file_sha256(Path(__import__("capped_votes").__file__)))
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"Verified {case} across {len(records)} budgets", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score"))
    parser.add_argument("official_npc", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--task", type=int, choices=range(len(CASES)))
    args = parser.parse_args()
    graph_path = args.output / "geometry.npz"
    if args.phase == "prepare":
        build_graph(args.official_npc / "inputs", graph_path)
    else:
        condition, seed = CASES[args.task]
        path = args.output / f"{condition}_s{seed}.json"
        score(args.official_npc, graph_path, condition, seed, path)


if __name__ == "__main__":
    main()
