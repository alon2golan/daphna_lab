"""Load verified CIFAR-100 experiment counts and form matched comparisons."""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CONDITIONS = {
    "sym30": "Symmetric 30%", "sym40": "Symmetric 40%", "sym50": "Symmetric 50%",
    "asym30": "Asymmetric 30%", "asym40": "Asymmetric 40%", "asym50": "Asymmetric 50%",
    "cifar100n": "CIFAR-100N", "clean": "Clean",
}
CHANNELS = {"uniform": "Uniform wrong classes", "scattered_cycle": "Fixed class cycle",
            "superclass_cycle": "Within-superclass cycle"}
RAW_DEPTHS = ["raw3", "raw4", "raw6", "raw8"]
GEOMETRY_SHA = "486509610f2947676c2bf5dfc7d3e6950e79805580290f9ca0d54790058e682b"
HUMAN_SHA = "fbe91e5598e2311cdd78a17f0123c76b4000446c044027f5584f096597fe5319"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confirmation_scores():
    folder = ROOT / "confirmation"
    orders_file = ROOT / "exact_backup" / "orders.json"
    orders = read(orders_file)
    exact_orders = {arm: orders[arm] for arm in ("maxherding", "backup_coverage")}
    if (exact_orders["maxherding"][:100] != exact_orders["backup_coverage"][:100]
            or orders["selector_sha256"] != sha(orders_file.with_name("dense_backup.py"))):
        raise ValueError("Exact acquisition archive changed")
    order_hash, source_hash = sha(orders_file), sha(folder / "evaluate.py")
    synthetic = [name for name in CONDITIONS if name.startswith(("sym", "asym"))]
    records = []
    for seed in (*range(25, 30), *range(40, 50)):
        result = read(folder / "results" / f"seed_{seed}.json")
        fixed_order = seed < 40
        budgets = (400, 800, 1200, 1600, 2000) if fixed_order else (400, 800, 1200, 1600)
        expected_conditions = list(CONDITIONS) if seed == 25 or not fixed_order else synthetic
        expected_acquisitions = exact_orders if fixed_order else {"seeded_coverage": None}
        if (result["seed"] != seed or result["test_count"] != 10000
                or result["geometry_sha256"] != GEOMETRY_SHA or result["human_sha256"] != HUMAN_SHA
                or result["source_sha256"] != source_hash
                or result["exact_orders_sha256"] != (order_hash if fixed_order else None)
                or (fixed_order and result["acquisition_rows"] != exact_orders)):
            raise ValueError(f"Invalid confirmation provenance in seed {seed}")
        expected = {(a, b, c) for a in expected_acquisitions for b in budgets for c in expected_conditions}
        actual = {(r["acquisition"], r["budget"], r["condition"]) for r in result["rows"]}
        if actual != expected or len(result["rows"]) != len(expected):
            raise ValueError(f"Incomplete confirmation in seed {seed}")
        for row in result["rows"]:
            record = {key: value for key, value in row.items() if key != "correct"}
            records.append(dict(record, seed=seed, samples=10000, **row["correct"]))
    return pd.DataFrame.from_records(records)


def sparse_scores():
    folder = ROOT / "sparse_backup"
    source_hash = sha(folder / "backup_coverage.py")
    records = []
    for seed in range(30, 35):
        result = read(folder / "results" / f"seed_{seed}.json")
        orders = result["acquisitions"]
        if (result["source_sha256"] != source_hash or result["geometry_sha256"] != GEOMETRY_SHA
                or orders["max"][:100] != orders["backup"][:100] or len(result["records"]) != 160):
            raise ValueError(f"Invalid sparse confirmation in seed {seed}")
        for row in result["records"]:
            records.append({
                "seed": row["seed"], "acquisition": row["arm"], "condition": row["condition"],
                "budget": row["budget"], "samples": row["test_count"], "depth": row["depth"],
                "correct": row["test_correct"], "1nn": row["nn_correct"],
            })
    scores = pd.DataFrame.from_records(records)
    keys = ["seed", "acquisition", "condition", "budget", "samples"]
    graph = scores.pivot(index=keys, columns="depth", values="correct")
    nearest = scores.loc[scores["depth"].eq(3)].set_index(keys)["1nn"]
    return graph.rename(columns={3: "raw3", 6: "raw6"}).join(nearest).reset_index()


def collision_scores():
    folder = ROOT / "confirmation"
    source_hash = sha(folder / "collision.py")
    channel_hash = sha(ROOT.parent / "source_support_study" / "channel.py")
    records = []
    for seed in range(40, 50):
        name = f"seed_{seed}.json"
        result = read(folder / "collision_results" / name)
        if (result["seed"] != seed or result["test_count"] != 10000
                or result["geometry_sha256"] != GEOMETRY_SHA or result["source_sha256"] != source_hash
                or result["channel_sha256"] != channel_hash
                or result["reference_sha256"] != sha(folder / "results" / name)):
            raise ValueError(f"Invalid corruption-channel provenance in seed {seed}")
        for row in result["rows"]:
            records.append(dict(seed=seed, samples=10000, condition=row["condition"], budget=row["budget"],
                                acquisition=row["acquisition"], **row["correct"]))
    return pd.DataFrame.from_records(records)


def vast_scores(stage, pool="unlabeled"):
    records = []
    folder = ROOT / "vast"
    for path in sorted((folder / "results" / stage).glob("*.json")):
        result = read(path)
        condition = "clean" if result["condition"] == "sym0" else result["condition"]
        record = dict(acquisition="protected", seed=result["seed"],
                      condition=condition, budget=result["budget"])
        if stage == "teacher":
            stats = result["rows"][pool]
            if result["reconstruction_error"] > 3e-6:
                raise ValueError(f"Teacher failed reconstruction in {path.name}")
            record.update(samples=stats["samples"], raw5=stats["raw"], sqrt5=stats["sqrt"])
        else:
            teacher = read(folder / "results" / "teacher" / path.name)
            if (result["source_sha256"] != sha(folder / "train_student.py") or result["epochs"] != 300
                    or result["original_target_sha256"] != teacher["original_sha256"]
                    or result["sqrt_target_sha256"] != teacher["revised_sha256"]):
                raise ValueError(f"Invalid student provenance in {path.name}")
            samples = result["test_count"]
            raw = result["archived_reference"]["test_accuracy_percent"]
            sqrt = result["sqrt"]["test_accuracy_percent"]
            record.update(samples=samples, raw5=round(raw * samples / 100), sqrt5=round(sqrt * samples / 100))
        records.append(record)
    return pd.DataFrame.from_records(records)


def compare(scores, candidate_acq, reference_acq, candidate_head, reference_head):
    """Match seeds and choose one baseline walk depth per cell when requested."""
    pairs = []
    for (condition, budget), cell in scores.groupby(["condition", "budget"], sort=False):
        candidate = cell.loc[cell["acquisition"].eq(candidate_acq)].set_index("seed").sort_index()
        reference = cell.loc[cell["acquisition"].eq(reference_acq)].set_index("seed").sort_index()
        if (candidate.index.empty or candidate.index.has_duplicates
                or not candidate["samples"].equals(reference["samples"])):
            raise ValueError(f"Unmatched seeds in {condition} at {budget} labels")
        head = reference[RAW_DEPTHS].mean().idxmax() if reference_head == "best_raw" else reference_head
        candidate_correct, reference_correct = candidate[candidate_head], reference[head]
        if (not candidate_correct.between(0, candidate["samples"]).all()
                or not reference_correct.between(0, reference["samples"]).all()):
            raise ValueError(f"Unexpected accuracy counts in {condition} at {budget} labels")
        delta = candidate_correct - reference_correct
        pairs.append(pd.DataFrame({
            "condition": condition, "budget": budget, "seed": candidate.index,
            "reference_head": head, "reference_pct": 100 * reference_correct / reference["samples"],
            "method_pct": 100 * candidate_correct / candidate["samples"],
            "delta_pp": 100 * delta / candidate["samples"], "win": delta > 0, "tie": delta == 0,
        }))
    return pd.concat(pairs, ignore_index=True)


def summarize(pairs):
    return pairs.groupby(["condition", "budget"]).agg(
        reference_head=("reference_head", "first"), reference_pct=("reference_pct", "mean"),
        method_pct=("method_pct", "mean"), delta_pp=("delta_pp", "mean"),
        wins=("win", "sum"), ties=("tie", "sum"), n=("seed", "size"),
    )
