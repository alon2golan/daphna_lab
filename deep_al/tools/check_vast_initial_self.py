"""Compare initialization-only unit self-evidence against saved VAST runs.

This diagnostic reuses each completed run's graph and selected labels. It does
not change the production initializer, repeated-update kernel, or train models.
"""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy import sparse

from deep_al.vast.data import file_hash
from deep_al.vast.inference import normalize_evidence, prepare_targets, propagate_literal, seed_evidence


def check_run(directory):
    directory = Path(directory).resolve()
    config = json.loads((directory / "config.json").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    original = json.loads((directory / "propagation.json").read_text())
    graph_info = json.loads((directory / "graph.json").read_text())
    kernel_path = Path(config["cache_dir"]) / (graph_info["cache_key"] + ".npz")
    kernel = sparse.load_npz(kernel_path)
    if np.any(kernel.diagonal() != 0):
        raise ValueError("The repeated-update kernel must have no self-loops")
    label_path = Path(provenance["files"]["train.labels"])
    if file_hash(label_path) != provenance["sha256"]["train.labels"]:
        raise ValueError("Training label file differs from the recorded run")
    labels = np.load(label_path, allow_pickle=False)
    selected = np.load(directory / "labeled_indices.npy", allow_pickle=False)
    classes = len(provenance["class_ids"])
    if labels.shape != (kernel.shape[0],) or len(np.unique(selected)) != len(selected):
        raise ValueError("Invalid label count or repeated selected sample IDs")
    mask = np.zeros(len(labels), dtype=bool)
    mask[selected] = True
    initial = seed_evidence(kernel, selected, labels[selected], classes)
    baseline_max = float(normalize_evidence(initial).max())
    if not np.isclose(baseline_max, original["initial_max_confidence"], atol=1e-12, rtol=1e-12):
        raise ValueError("Reconstructed initialization differs from the saved run")

    # Raw RBF self-similarity is one. Add it once to the known class only;
    # neither graph degrees nor the propagation kernel are modified.
    initial[selected, labels[selected]] += 1.0
    q = normalize_evidence(initial)
    before_max = float(q.max())
    labeled_confidence = q[selected].max(axis=1)
    labeled_stats = {"min": float(labeled_confidence.min()),
                     "median": float(np.median(labeled_confidence)),
                     "max": float(labeled_confidence.max())}
    del q
    evidence, info = propagate_literal(kernel, initial, mask)
    np.testing.assert_array_equal(evidence[selected], initial[selected])
    if info["history"][-1]["max_confidence"] > before_max + 1e-12:
        raise AssertionError("Positive mixing unexpectedly increased the global maximum")
    if np.any(kernel.diagonal() != 0):
        raise AssertionError("Diagnostic modified the propagation kernel")
    _, _, admitted, gate = prepare_targets(evidence, mask)
    return {"source_run": str(directory), "dataset": config["dataset"],
            "labels_per_class": config["labels_per_class"], "seed": config["seed"],
            "representation": provenance["representation"],
            "source_git_revision": provenance["local_git_revision"],
            "graph_cache_key": graph_info["cache_key"],
            "labeled_indices_sha256": file_hash(directory / "labeled_indices.npy"),
            "train_labels_sha256": provenance["sha256"]["train.labels"],
            "self_evidence_weight": 1.0, "self_evidence_stage": "initialization_only",
            "density_normalization": "off-diagonal graph unchanged; unit self contribution",
            "alpha0": 1.0 / classes, "freezing_threshold": 0.75,
            "baseline_initial_max_confidence": baseline_max,
            "initial_max_confidence": before_max,
            "initial_labeled_confidence": labeled_stats,
            "propagation": info, "admitted_count": int(admitted.sum()),
            "gate_base_threshold": gate["base_threshold"],
            "classifier_trained": False, "confirmed_reproduction": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = []
    for root in args.run_root:
        for directory in sorted(root.glob("*_labels*_seed*")):
            if (directory / "result.json").is_file():
                result = check_run(directory)
                results.append(result)
                print(json.dumps(result), flush=True)
    if not results:
        raise ValueError("No completed runs found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"diagnostic": "initialization-only unit self-evidence",
                                      "runs": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
