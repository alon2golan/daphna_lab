"""Compare early rounds of exact author code with the literal-paper diagnostic.

The author-code settings below are a controlled diagnostic, not a recovered
paper experiment configuration. No classifier is trained or output overwritten.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy import sparse

from deep_al.vast import author_reference
from deep_al.vast.data import file_hash
from deep_al.vast.inference import normalize_evidence, propagate_literal, seed_evidence


def author_kwargs(selected, labels, mask, classes):
    return dict(labeled_mask=mask, source_indices=selected, source_labels=labels,
                alpha=1 / classes, sender_mode="high_confidence",
                sender_threshold=1 / classes, receiver_mode="high_confidence",
                receiver_threshold=.75, normalization="sum", decay_rate=1.0)


def audit(initial, kernel, selected, labels, mask, rounds):
    classes = initial.shape[1]
    kwargs = author_kwargs(selected, labels, mask, classes)
    history = []
    current = initial.copy()
    previous_frozen = mask | (normalize_evidence(current).max(axis=1) > .75)
    for iteration in range(rounds + 1):
        q = normalize_evidence(current)
        frozen = mask | (q.max(axis=1) > .75)
        eligible = author_reference.compute_sender_mask(
            current, mask, mode="high_confidence", threshold=1 / classes)
        informed = current.max(axis=1) > current.min(axis=1)
        history.append({"round": iteration, "max_confidence": float(q.max()),
                        "max_unlabeled_confidence": float(q[~mask].max()),
                        "mean_unlabeled_confidence": float(q[~mask].max(axis=1).mean()),
                        "uninformed_unlabeled": int(((~mask) & ~informed).sum()),
                        "eligible_unlabeled_senders": int(eligible.sum()),
                        "prior_only_eligible_senders": int((eligible & ~informed).sum()),
                        "frozen_unlabeled": int((frozen & ~mask).sum()),
                        "newly_frozen": int((frozen & ~previous_frozen).sum())})
        previous_frozen = frozen
        if iteration < rounds:
            with contextlib.redirect_stdout(io.StringIO()):
                current = author_reference.propagate_labels(
                    current, kernel, num_iterations=1, **kwargs)
    # At decay=1, sequential single-round calls must equal the exact original
    # multi-round function, including persistent freezing and restored sources.
    with contextlib.redirect_stdout(io.StringIO()):
        exact = author_reference.propagate_labels(initial, kernel, num_iterations=rounds, **kwargs)
    np.testing.assert_allclose(current, exact, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(current[mask], initial[mask])
    del exact
    literal, literal_info = propagate_literal(kernel, initial, mask)
    literal_q = normalize_evidence(literal)
    return {"author": {"settings": {k: v for k, v in kwargs.items()
                                      if k not in ("labeled_mask", "source_indices", "source_labels")},
                        "rounds": rounds, "history": history,
                        "sequential_vs_original_parity": True},
            "literal_paper": {**literal_info,
                              "max_unlabeled_confidence": float(literal_q[~mask].max())}}


def check_run(directory, rounds):
    directory = Path(directory).resolve()
    cfg = json.loads((directory / "config.json").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    graph_info = json.loads((directory / "graph.json").read_text())
    kernel_path = Path(cfg["cache_dir"]) / (graph_info["cache_key"] + ".npz")
    kernel = sparse.load_npz(kernel_path)
    if np.any(kernel.diagonal()):
        raise ValueError("Expected the saved zero-diagonal graph")
    label_path = Path(provenance["files"]["train.labels"])
    if file_hash(label_path) != provenance["sha256"]["train.labels"]:
        raise ValueError("Training labels differ from original run")
    labels = np.load(label_path, allow_pickle=False)
    selected = np.load(directory / "labeled_indices.npy", allow_pickle=False)
    mask = np.zeros(len(labels), dtype=bool)
    mask[selected] = True
    initial = seed_evidence(kernel, selected, labels[selected], len(provenance["class_ids"]))
    saved = json.loads((directory / "propagation.json").read_text())
    np.testing.assert_allclose(normalize_evidence(initial).max(), saved["initial_max_confidence"],
                               rtol=1e-12, atol=1e-12)
    result = audit(initial, kernel, selected, labels[selected], mask, rounds)
    return {"source_run": str(directory), "dataset": cfg["dataset"],
            "labels_per_class": cfg["labels_per_class"], "seed": cfg["seed"],
            "representation": provenance["representation"],
            "kernel_sha256": file_hash(kernel_path),
            "selected_indices_sha256": file_hash(directory / "labeled_indices.npy"),
            "initialization": "same paper sum-plus-prior initializer for both recurrences; no self-loops",
            **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, action="append", default=[])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1 or args.output.exists():
        raise ValueError("Use positive rounds and a fresh output path")
    if args.run_root:
        results = []
        for root in args.run_root:
            for directory in sorted(root.glob("*_labels*_seed*")):
                if (directory / "result.json").is_file():
                    result = check_run(directory, args.rounds)
                    results.append(result)
                    print(json.dumps(result), flush=True)
        if not results:
            raise ValueError("No completed runs found")
    else:
        kernel = sparse.diags([np.ones(4), np.ones(4)], [-1, 1], shape=(5, 5), format="csr")
        selected, labels = np.array([0]), np.array([0])
        mask = np.arange(5) == 0
        initial = seed_evidence(kernel, selected, labels, 100)
        results = [{"dataset": "synthetic five-point chain, 100 possible classes",
                    **audit(initial, kernel, selected, labels, mask, args.rounds)}]
    record = {"author_commit": author_reference.SOURCE_COMMIT,
              "author_source": author_reference.SOURCE_PATH,
              "author_reference_sha256": file_hash(author_reference.__file__),
              "literal_inference_sha256": file_hash(Path(__file__).resolve().parents[1] / "vast" / "inference.py"),
              "audit_script_sha256": file_hash(__file__),
              "confirmed_reproduction": False, "classifier_trained": False,
              "runs": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Audit saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
