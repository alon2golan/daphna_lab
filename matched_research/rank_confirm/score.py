"""Score raw and AUM-rank walks on the frozen outputs of each released NPC run."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import rank_weights
import replay
import run_official
from run_fresh import CASES, GRAPH_SHA, WRAPPER_SHA


RAW_DEPTHS = (3, 4, 5, 6, 7, 8)
REPLAY_SHA = "a1e6b0338511276b515223c054ee246e8c5f8c79faaddd3407cc3521b5637b95"
RANK_SHA = "f022f5f12e701b853c3df928a4b220586e0930ad2796ec26a12cc941ae75cca3"


def read(path):
    return json.loads(path.read_text())


def verify_checksum(path, expected):
    actual = run_official.file_sha256(path)
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {path}: {actual}")


def raw_depth_predictions(walk, project, rows, labels):
    """Aggregate by class before propagating to verify raw depth six independently."""
    evidence = np.zeros((walk.shape[0], 100), dtype=np.float64)
    evidence[rows, labels] = 1
    for depth in range(1, max(RAW_DEPTHS) + 1):
        if depth in RAW_DEPTHS:
            yield depth, (project @ evidence).argmax(axis=1)
        if depth < max(RAW_DEPTHS):
            evidence = walk @ evidence


def stock_results(run, condition, seed, test_count):
    summary = read(run / "summary.json")
    provenance = read(run / "provenance.json")
    identity = (condition, seed, False)
    if (any((row["condition"], row["seed"], row["smoke"]) != identity
            for row in (summary, provenance))
            or provenance["upstream_commit"] != run_official.UPSTREAM_COMMIT
            or provenance["upstream_source_sha256"] != run_official.UPSTREAM_SOURCE_SHA256
            or provenance["wrapper_sha256"] != WRAPPER_SHA
            or provenance["input_manifest"]["bank_sha256"] != run_official.BANK_HASHES
            or "RTX 2080" not in provenance["gpu"]):
        raise ValueError(f"Official NPC configuration changed: {run}")
    cli = provenance["official_cli"]
    flags = dict(zip(cli[:-1], cli[1:]))
    expected = {
        "--al": "npc", "--initial_budget_size": "0", "--budget": "100",
        "--num_episodes": "20", "--distance_metric": "l2",
        "--features_normalization": "unit_norm", "--initial_delta": "0.65",
        "--delta_policy": "constant", "--lnl": "aum", "--max_epoch": "500",
        "--noise_mom_coeff": "0", "--seed": str(seed),
    }
    if (any(flags.get(key) != value for key, value in expected.items())
            or "--use_linear_model" not in cli or "--use_noise_dropout" in cli
            or "--train_on_all_labeled_data" in cli):
        raise ValueError(f"Official NPC command changed: {run}")
    records = summary["cells"]
    if tuple(row["budget"] for row in records) != run_official.BUDGETS:
        raise ValueError(f"Missing official checkpoints: {run}")
    for row in records:
        folder = run / "fits" / f"{row['budget']:05d}"
        if read(folder / "baseline_result.json") != row:
            raise ValueError(f"Official baseline summary and fit differ: {folder}")
        verify_checksum(folder / "fit_inputs.npz", row["fit_inputs_sha256"])
        hits = round(row["accuracy"] * test_count / 100)
        if (not 0 <= hits <= test_count
                or row["accuracy"] != 100 * hits / test_count
                or any(not re.fullmatch("[0-9a-f]{64}", row[key])
                       for key in ("prediction_sha256", "fitted_model_sha256"))):
            raise ValueError(f"Invalid official held-out test result: {folder}")
        yield row, hits


def score(condition, seed, runs, graph_path, truth_path, output, parity=None):
    for module, digest in ((run_official, WRAPPER_SHA), (rank_weights, RANK_SHA),
                           (replay, REPLAY_SHA)):
        verify_checksum(Path(module.__file__), digest)
    verify_checksum(graph_path, GRAPH_SHA)
    verify_checksum(truth_path, run_official.BANK_HASHES["test_targets"])
    truth = np.load(truth_path, allow_pickle=False)
    if truth.shape != (10000,) or np.any((truth < 0) | (truth >= 100)):
        raise ValueError("Test targets must match 10,000 canonical CIFAR-100 rows")
    with np.load(graph_path, allow_pickle=False) as geometry:
        neighbors = geometry["neighbors"]
        heldout = geometry["heldout"]
        bank = json.loads(geometry["bank"].item())
    if bank != run_official.BANK_HASHES or (neighbors.shape, heldout.shape) != ((50000, 13), (10000, 13)):
        raise ValueError("Graph and released feature inputs differ")

    walk = replay.path_matrix(neighbors, len(neighbors))
    project = replay.path_matrix(heldout, len(neighbors))
    name = f"{condition}_s{seed}"
    run = runs / name
    stock = list(stock_results(run, condition, seed, len(truth)))
    with np.load(run / "fits/02000/fit_inputs.npz", allow_pickle=False) as fit:
        anchors = fit["acquired_indices"]
        observed = fit["observed_labels"]
    if (anchors.shape != (2000,) or len(np.unique(anchors)) != len(anchors)
            or np.any((anchors < 0) | (anchors >= len(neighbors)))):
        raise ValueError(f"Invalid final acquisition order: {run}")
    paths = replay.six_edge_counts(walk, project, anchors)
    budget_rows = []
    for base, official_count in stock:
        budget = base["budget"]
        with np.load(run / "fits" / f"{budget:05d}" / "fit_inputs.npz", allow_pickle=False) as fit:
            np.testing.assert_array_equal(fit["acquired_indices"], anchors[:budget])
            np.testing.assert_array_equal(fit["observed_labels"], observed[:budget])
            keep, aum = fit["base_keep"], fit["aum_scores"]
        if (keep.shape != (budget,) or keep.dtype != np.bool_ or int(keep.sum()) != base["kept"]
                or not keep.any() or aum.shape != keep.shape or not np.isfinite(aum).all()):
            raise ValueError(f"Invalid official keep mask or AUM scores: {name}/{budget}")
        labels = observed[:budget][keep]
        weights = rank_weights.class_weights(labels, aum[keep])
        np.testing.assert_allclose(np.bincount(labels, weights=weights), np.bincount(labels),
                                   atol=1e-10, rtol=0)
        accepted = paths[:, :budget][:, keep]
        raw6 = rank_weights.predict(accepted, labels)
        rank6 = rank_weights.predict(accepted * weights, labels)
        predictions = dict(raw_depth_predictions(walk, project, anchors[:budget][keep], labels))
        np.testing.assert_array_equal(predictions[6], raw6)
        raw_counts = {f"raw{depth}": int((pred == truth).sum()) for depth, pred in predictions.items()}
        budget_rows.append(dict(
            budget=budget, retained=int(keep.sum()), official=official_count, rank6=int((rank6 == truth).sum()),
            **raw_counts,
            input_sha256=base["fit_inputs_sha256"],
            official_prediction_sha256=base["prediction_sha256"],
            raw6_prediction_sha256=run_official.array_sha256(raw6.astype("<i8")),
            rank6_prediction_sha256=run_official.array_sha256(rank6.astype("<i8")),
        ))
    if parity:
        for new, prior in zip(budget_rows, read(parity)["rows"], strict=True):
            for new_key, prior_key in (("budget", "budget"), ("official", "official"),
                                       ("raw6", "raw"), ("rank6", "rank"),
                                       ("input_sha256", "input_sha256"),
                                       ("official_prediction_sha256", "official_prediction_sha256")):
                if new[new_key] != prior[prior_key]:
                    raise ValueError(f"Historical parity failed for {name}/{new['budget']} field {new_key}")

    result = dict(
        condition=condition, seed=seed, test_count=len(truth), rows=budget_rows,
        graph_sha256=GRAPH_SHA, wrapper_sha256=WRAPPER_SHA, rank_sha256=RANK_SHA,
        source_sha256=run_official.file_sha256(Path(__file__)),
        stock_summary_sha256=run_official.file_sha256(run / "summary.json"),
        stock_provenance_sha256=run_official.file_sha256(run / "provenance.json"),
    )
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"Verified {len(budget_rows)} held-out pairs for {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("official", type=Path)
    parser.add_argument("graph", type=Path)
    parser.add_argument("task", type=int, choices=range(len(CASES)))
    parser.add_argument("--parity", type=Path)
    args = parser.parse_args()
    condition, seed = ("clean", 71) if args.parity else CASES[args.task]
    runs = (args.official if args.parity else args.work) / "runs"
    output = args.work / ("parity.json" if args.parity else f"results/{condition}_s{seed}.json")
    score(condition, seed, runs, args.graph, args.official / "inputs/test_targets.npy",
          output, args.parity)


if __name__ == "__main__":
    main()
