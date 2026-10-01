"""Single-seed CLI and explicit nine-setting grid."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy
import torch

from . import SOURCE_COMMIT
from .data import DATASETS, file_hash, load_manifest, read_config, select_labeled
from .graph import cached_kernel
from .inference import require_propagation, seed_evidence, propagate_literal, prepare_targets
from .source_propagation import propagate_source
from .training import train_classifier, predict
from .trace import PropagationTrace

DEFAULTS = {"seed": 0, "epochs": 300, "device": "auto", "backend": "faiss",
            "max_neighbors": 512, "max_rounds": None, "propagation": "unresolved",
            "cache_dir": None, "cpu_threads": 1,
            "sender_threshold": .05, "receiver_threshold": .75}
CONFIG_KEYS = set(DEFAULTS) | {"batch_size", "distillation_weight"}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def config_for(args, dataset):
    settings = DATASETS[dataset]
    cfg = {**DEFAULTS, "batch_size": settings["batch_size"], "distillation_weight": settings["weight"]}
    if args.config:
        supplied = read_config(args.config)
        unknown = set(supplied) - CONFIG_KEYS
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        cfg.update(supplied)
    for key in CONFIG_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    for key in ("seed", "epochs", "batch_size", "max_neighbors", "cpu_threads"):
        if type(cfg[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if min(cfg["epochs"], cfg["max_neighbors"], cfg["cpu_threads"]) < 1 or cfg["batch_size"] < 2:
        raise ValueError("epochs/neighbors/threads must be positive and batch_size >= 2")
    if cfg["max_rounds"] is not None and (type(cfg["max_rounds"]) is not int or cfg["max_rounds"] < 1):
        raise ValueError("max_rounds must be a positive integer")
    if not isinstance(cfg["distillation_weight"], (int, float)) or not np.isfinite(cfg["distillation_weight"]) or cfg["distillation_weight"] < 0:
        raise ValueError("distillation_weight must be finite and nonnegative")
    if cfg["backend"] not in ("dense", "faiss") or cfg["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("Invalid backend/device")
    if cfg["propagation"] not in ("unresolved", "literal-paper", "author-source"):
        raise ValueError("No confirmed propagation mode is available")
    for key in ("sender_threshold", "receiver_threshold"):
        if not isinstance(cfg[key], (int, float)) or not np.isfinite(cfg[key]):
            raise ValueError(f"{key} must be finite")
    if not (0 <= cfg["sender_threshold"] < 1 and 0 < cfg["receiver_threshold"] <= 1):
        raise ValueError("Invalid propagation thresholds")
    if cfg["propagation"] == "author-source" and cfg["max_rounds"] is None:
        cfg["max_rounds"] = 50
    if cfg["cache_dir"] is not None and not isinstance(cfg["cache_dir"], str):
        raise ValueError("cache_dir must be a path string")
    return cfg


def parser():
    p = argparse.ArgumentParser(description="VAST: one seed per run; propagation unconfirmed")
    p.add_argument("--manifest", required=True)
    p.add_argument("--labels-per-class", type=int, choices=(1, 2, 4), required=True)
    p.add_argument("--config")
    p.add_argument("--output-dir")
    p.add_argument("--preflight", action="store_true", help="Validate manifest/config without running or writing outputs")
    for key in ("seed", "epochs", "batch_size", "max_neighbors", "max_rounds", "cpu_threads"):
        p.add_argument("--" + key.replace("_", "-"), type=int)
    p.add_argument("--distillation-weight", type=float)
    p.add_argument("--sender-threshold", type=float)
    p.add_argument("--receiver-threshold", type=float)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"))
    p.add_argument("--backend", choices=("faiss", "dense"))
    p.add_argument("--cache-dir")
    p.add_argument("--propagation", choices=("unresolved", "literal-paper", "author-source"),
                   help="Explicit experimental protocol; neither mode confirms numerical reproduction")
    return p


def run(args):
    # Resolve configuration before loading large files or creating output directories.
    meta = read_config(args.manifest)
    if meta.get("dataset") not in DATASETS:
        raise ValueError("Manifest dataset is unsupported")
    config = config_for(args, meta["dataset"])
    if not args.preflight:
        require_propagation(config["propagation"])
        if not args.output_dir:
            raise ValueError("--output-dir is required for a run")
        if Path(args.output_dir).exists():
            raise ValueError("Output directory already exists; choose a new directory")
    data = load_manifest(args.manifest, args.labels_per_class)
    if args.preflight:
        summary = {"preflight": "passed", "dataset": data.dataset,
                   "train_count": len(data.train_features), "test_count": len(data.test_features),
                   "input_width": data.train_features.shape[1], "seed": config["seed"],
                   "propagation": config["propagation"], "confirmed_reproduction": False}
        print(json.dumps(summary))
        return summary
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "running", "confirmed_reproduction": False})
    try:
        print(f"EXPERIMENTAL {config['propagation']} mode; not confirmed paper reproduction.", flush=True)
        config.update({"dataset": data.dataset, "labels_per_class": args.labels_per_class,
                       "source_commit": SOURCE_COMMIT, "classifier_input": "raw",
                       "confirmed_reproduction": False})
        write_json(output / "config.json", config)
        provenance = {**data.provenance, "representation": data.representation,
                      "python": platform.python_version(), "numpy": np.__version__,
                      "scipy": scipy.__version__, "torch": torch.__version__,
                      "platform": platform.platform(), "source_commit": SOURCE_COMMIT,
                      "faiss_threads": int(os.environ.get("VAST_FAISS_THREADS", min(os.cpu_count() or 1, 8))),
                      "code_sha256": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
        try:
            provenance["local_git_revision"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            provenance["local_git_revision"] = None
        write_json(output / "provenance.json", provenance)
        labeled = select_labeled(data.train_labels, data.classes, args.labels_per_class, config["seed"])
        np.save(output / "labeled_indices.npy", labeled)
        np.save(output / "labeled_sample_ids.npy", data.train_ids[labeled])
        labeled_mask = np.zeros(len(data.train_features), dtype=bool)
        labeled_mask[labeled] = True
        kernel, graph_info = cached_kernel(data.train_features, config["backend"],
                                           config["max_neighbors"], config["cache_dir"])
        write_json(output / "graph.json", graph_info)
        initial = seed_evidence(kernel, labeled, data.train_labels[labeled], data.classes)
        trace = PropagationTrace(output / "propagation_trace", data.train_features,
                                 data.train_ids, labeled_mask, data.provenance["class_ids"])
        trace.metadata["propagation_mode"] = config["propagation"]
        if config["propagation"] == "author-source":
            evidence, propagation_info = propagate_source(
                kernel, initial, labeled_mask, labeled, data.train_labels[labeled],
                config["max_rounds"], config["sender_threshold"], config["receiver_threshold"],
                observer=trace)
        else:
            evidence, propagation_info = propagate_literal(kernel, initial, labeled_mask,
                                                           config["max_rounds"], observer=trace)
        initial_max = float((initial / initial.sum(axis=1, keepdims=True)).max())
        if config["propagation"] == "literal-paper":
            edge_bound = 1 / np.sqrt(np.log(2))
            mass = args.labels_per_class * edge_bound
            propagation_info["theoretical_initial_confidence_ceiling"] = (1 / data.classes + mass) / (1 + mass)
        propagation_info["initial_max_confidence"] = initial_max
        if config["propagation"] == "author-source":
            propagation_info["sender_threshold_provenance"] = (
                "0.05 is reported in Itai's inspected CIFAR100 run log; reused for other datasets as an experiment"
            )
        write_json(output / "propagation.json", propagation_info)
        targets, confidence, admitted, gate_info = prepare_targets(evidence, labeled_mask)
        write_json(output / "gate.json", gate_info)
        np.save(output / "admitted_indices.npy", np.flatnonzero(admitted))
        np.savez_compressed(output / "soft_targets.npz", probabilities=targets, confidence=confidence)
        # Free graph/evidence before training the classifier.
        del kernel, initial, evidence
        model, device = train_classifier(data.train_features, labeled, data.train_labels[labeled],
                                         data.classes, targets, confidence, admitted, config, output)
        predictions = predict(model, data.test_features, device)
        np.save(output / "test_predictions.npy", predictions)
        # Export evaluation metadata only after training/prediction, so copied
        # run folders support confusion matrices without the original data bank.
        np.save(output / "test_labels.npy", data.test_labels)
        np.save(output / "test_sample_ids.npy", data.test_ids)
        accuracy = float(100 * np.mean(predictions == data.test_labels))
        reference = DATASETS[data.dataset]["reference"][args.labels_per_class]
        result = {"dataset": data.dataset, "labels_per_class": args.labels_per_class,
                  "seed": config["seed"], "representation": data.representation["name"],
                  "test_accuracy_percent": accuracy, "paper_mean_reference": reference,
                  "difference_from_paper_mean": accuracy - reference, "seed_count": 1,
                  "sd": None, "sem": None, "epochs": config["epochs"], "device": device,
                  "labeled_only": not bool(admitted.any()), "propagation": config["propagation"],
                  "confirmed_reproduction": False, "status": "completed_experimental"}
        write_json(output / "result.json", result)
        write_json(output / "status.json", {"status": "completed_experimental", "confirmed_reproduction": False})
        print(json.dumps(result), flush=True)
        return result
    except BaseException as exc:
        write_json(output / "status.json", {"status": "failed", "error": str(exc), "confirmed_reproduction": False})
        raise


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        run(args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        p.exit(2, f"VAST: {exc}\n")


def grid_commands(manifests, output_root, seed=0, propagation="unresolved", epochs=300, device="auto"):
    script = Path(__file__).resolve().parents[1] / "tools" / "train_vast.py"
    commands = []
    seen = set()
    for path in manifests:
        spec = read_config(path)
        dataset = spec["dataset"]
        if dataset not in DATASETS or dataset in seen:
            raise ValueError("Grid needs exactly one manifest for each of the three datasets")
        seen.add(dataset)
        for budget in (1, 2, 4):
            dest = Path(output_root) / f"{dataset}_labels{budget}_seed{seed}"
            commands.append([sys.executable, str(script), "--manifest", str(Path(path).resolve()),
                             "--labels-per-class", str(budget), "--seed", str(seed),
                             "--epochs", str(epochs), "--propagation", propagation, "--device", device,
                             "--cache-dir", str(Path(output_root) / "graph_cache"),
                             "--output-dir", str(dest)])
    if seen != set(DATASETS):
        raise ValueError("Grid requires all three datasets")
    return commands


def grid_main(argv=None):
    p = argparse.ArgumentParser(description="Nine VAST settings at one seed, sequential execution")
    p.add_argument("--manifests", nargs=3, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--propagation", choices=("unresolved", "literal-paper", "author-source"), default="unresolved")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    try:
        if not 0 <= args.seed < 2**32 or args.epochs < 1:
            raise ValueError("Invalid seed or epoch count")
        commands = grid_commands(args.manifests, args.output_dir, args.seed, args.propagation, args.epochs, args.device)
        if not args.dry_run:
            require_propagation(args.propagation)
        # All datasets and output destinations checked before any run starts.
        for path in args.manifests:
            load_manifest(path, labels_per_class=4)
        if any(Path(command[-1]).exists() for command in commands):
            raise ValueError("A grid output already exists; use a fresh output directory")
        if args.dry_run:
            print(json.dumps({"commands": commands, "run_count": len(commands), "seed": args.seed}, indent=2))
            return
        results = []
        for command in commands:
            subprocess.run(command, check=True)
            results.append(json.loads((Path(command[-1]) / "result.json").read_text()))
            with (Path(args.output_dir) / "summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        p.exit(2, f"VAST grid: {exc}\n")
