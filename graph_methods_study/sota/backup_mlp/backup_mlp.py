"""Transfer exact Gaussian backup coverage to the archived classifier protocol."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from deep_al.vast_noisy.herding20261119 import evaluate, inputs, study, worker
from deep_al.vast_noisy.submission20261019 import run as source

CONDITIONS = ("clean", "sym30", "sym50", "asym30", "asym50")
BUDGETS = (800, 1200, 2000)
HERDING = ("maxherding", "uherding", "js")
FOLLOWUP = ("observed_10", "soft_10")


def prepare(root, archive):
    plan = worker.checked_plan(archive)
    for name in inputs.COMMON_FILES:
        path = root / "inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive / "inputs" / name, path)
        if source.a.manifests.file_hash(path) != plan["input_sha256"][name]:
            raise ValueError(f"Input failed verification: {name}")
    return plan


def archived_score(base, budget, plan, pool):
    path = base / f"budget_{budget:04d}" / "metrics.json"
    metrics = source.read(path)
    prepared = base / f"inputs_{budget:04d}"
    manifest = source.read(prepared / "preparation.json")
    ids = np.load(prepared / "partitions" / "labeled_indices.npy", allow_pickle=False)
    labels = np.load(prepared / "observed_labeled_labels.npy", allow_pickle=False)
    original = Path(metrics["classifier_result"])
    proof = source.read(original.with_name("verification.json"))
    if (metrics["annotation_budget"] != budget or metrics["full_test_count"] != 10000
            or metrics["membership_key"] != study.membership_key(ids, labels)
            or manifest["config_sha256"] != plan["input_sha256"]["config.yaml"]
            or manifest["seed"] != plan["seed"]
            or proof["optimizer_steps"] != plan["epochs"] * budget // study.BATCH
            or proof["full_test_count"] != 10000
            or proof["recomputed_accuracy_percent"] != metrics["classifier_accuracy_percent"]):
        raise ValueError(f"Archived comparison does not match its protocol: {path}")
    np.testing.assert_array_equal(labels, pool[ids])
    for name in inputs.VALIDATION_FILES:
        if source.a.manifests.file_hash(prepared / name) != plan["input_sha256"][name]:
            raise ValueError("Archive validation rows changed")
    return {"mlp": metrics["classifier_accuracy_percent"], "1nn": metrics["accuracy_1nn_percent"],
            "metrics_sha256": source.a.manifests.file_hash(path), "membership_key": metrics["membership_key"]}


def compare(index, root, archive, js, orders_file):
    plan = worker.checked_plan(archive)
    condition = CONDITIONS[index]
    for name in inputs.COMMON_FILES:
        if source.a.manifests.file_hash(root / "inputs" / name) != plan["input_sha256"][name]:
            raise ValueError(f"Private input changed: {name}")
    arrays, eligible, _ = worker.load_data(archive, plan)
    orders = source.read(orders_file)
    if orders["common_prefix"] != 100 or orders["backup_weight"] != .5:
        raise ValueError("Wrong backup coverage trajectory")
    selected = np.asarray(orders["backup_coverage"])
    archive_start = np.load(archive / "trajectories" / "clean_maxherding" / "selected_indices.npy",
                            allow_pickle=False)
    np.testing.assert_array_equal(selected[:100], archive_start[:100])
    if not np.isin(selected, eligible).all() or len(np.unique(selected)) != max(BUDGETS):
        raise ValueError("Invalid backup query rows")
    noise = plan["noises"][condition]
    noise_file = Path(noise["observed_path"])
    if source.a.manifests.file_hash(noise_file) != noise["observed_sha256"]:
        raise ValueError("Archived observations changed")
    pool = np.load(noise_file, allow_pickle=False)
    methods = [(m, archive) for m in HERDING] + [(m, js) for m in FOLLOWUP]
    task = dict(next(t for t in plan["tasks"] if t["condition"] == condition),
                id=f"{condition}_backup_coverage", method="backup_coverage")
    cases = []
    for budget in BUDGETS:
        scores = {m: archived_score(base / "trajectories" / f"{condition}_{m}", budget, plan, pool)
                  for m, base in methods}
        cases.append({"budget": budget, "scores": scores})
    for record in cases:
        budget = record["budget"]
        chosen = selected[:budget]
        remaining = np.setdiff1d(eligible, chosen)
        destination = root / "trajectories" / task["id"] / f"budget_{budget:04d}"
        trained = evaluate.checkpoint(root, destination, plan, task, arrays, chosen, pool[chosen],
                                      remaining, {})
        record["scores"]["backup_coverage"] = {
            "mlp": trained["classifier_accuracy_percent"], "1nn": trained["accuracy_1nn_percent"],
            "membership_key": trained["membership_key"],
        }
        print(condition, budget, record["scores"]["backup_coverage"]["mlp"], flush=True)
    packet = {"condition": condition, "seed": 0, "test_count": 10000,
              "epochs": plan["epochs"], "observed_sha256": noise["observed_sha256"],
              "source_sha256": source.a.manifests.file_hash(__file__),
              "orders_sha256": source.a.manifests.file_hash(orders_file), "cases": cases}
    path = root / f"{condition}.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("js", type=Path)
    parser.add_argument("orders", type=Path)
    parser.add_argument("--index", type=int, choices=range(len(CONDITIONS)))
    args = parser.parse_args()
    if args.index is None:
        prepare(args.root, args.archive)
    else:
        compare(args.index, args.root, args.archive, args.js, args.orders)
