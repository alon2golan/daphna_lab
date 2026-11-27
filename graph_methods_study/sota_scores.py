"""Verified counts for acquisition benchmarks and exploratory methods."""

import pandas as pd

from analyze import GEOMETRY_SHA, ROOT, read, sha

SOTA = ROOT / "sota"
ACQUISITIONS = {
    "backup_coverage": "Backup coverage",
    "npc_backup": "NPC + backup",
    "weighted_backup": "Weighted NPC + backup",
    "fragility": "Source fragility",
    "npc": "Netta NPC",
    "weighted_npc": "Netta Weighted NPC",
    "maxherding_nas": "MaxHerding + NAS",
    "probcover": "ProbCover",
    "maxherding": "MaxHerding",
    "uherding": "Lab U-Herding",
    "js": "JS-1",
    "observed_10": "JS-10, observed broadcast",
    "soft_10": "JS-10, soft broadcast",
}


def acquisition_scores():
    table, annotations = {}, {}

    def register(seed, condition, observed):
        if observed and condition != "cifar100n":
            previous = annotations.setdefault((seed, condition), observed)
            if previous != observed:
                raise ValueError(f"Different annotation vectors for {condition}, seed {seed}")

    def add(seed, condition, budget, method, training_filter, correct):
        method = "maxherding" if method == "maxherding_reference" else method
        for head, value in correct.items():
            key = (seed, condition, budget, method, training_filter, head)
            prior = table.setdefault(key, value)
            if not 0 <= value <= 10000 or prior != value:
                raise ValueError(f"Invalid matched count at {key}")

    archive = SOTA / "archive"
    for path in sorted((archive / "results").glob("condition_*.json")):
        result = read(path)
        if (result["geometry_sha256"] != GEOMETRY_SHA or result["source_sha256"] != sha(archive / "replay.py")
                or result["evaluator_sha256"] != sha(ROOT / "confirmation" / "evaluate.py")
                or result["test_count"] != 10000):
            raise ValueError(f"Archive provenance mismatch in {path.name}")
        register(result["seed"], result["condition"], result["observed_sha256"])
        for row in result["rows"]:
            add(result["seed"], row["condition"], row["budget"], row["acquisition"], "all", row["correct"])

    folder = SOTA / "fragility"
    for path in sorted((folder / "results").glob("case_*.json")):
        result = read(path)
        if (result["source_sha256"] != sha(folder / "fragility.py") or result["geometry_sha256"] != GEOMETRY_SHA
                or result["test_count"] != 10000):
            raise ValueError(f"Fragility provenance mismatch in {path.name}")
        register(result["seed"], result["condition"], result["labels_sha256"])
        for method, rows in result["arms"].items():
            for row in rows:
                add(result["seed"], row["condition"], row["budget"], method, "all", row["correct"])

    folder = SOTA / "netta"
    raw = read(folder / "results" / "scores.json")
    if raw["source_sha256"] != sha(folder / "netta_baselines.py") or len(raw["records"]) != 40:
        raise ValueError("NPC baseline scores fail provenance")
    for run in raw["records"]:
        source, condition = run["provenance"], run["condition"]
        if source["head_graph_sha256"] != GEOMETRY_SHA or source["source_sha256"] != raw["source_sha256"]:
            raise ValueError("NPC baseline geometry or source changed")
        register(source["seed"], condition, source["observed_sha256"])
        for row in run["scores"]:
            add(source["seed"], condition, row["budget"], run["method"], row["training_filter"], row["correct"])

    folder = SOTA / "npc_backup"
    for path in sorted((folder / "results").glob("case_*.json")):
        result = read(path)
        if (result["head_graph_sha256"] != GEOMETRY_SHA or result["test_count"] != 10000
                or result["source_sha256"] != sha(folder / "npc_backup.py")
                or result["netta_sha256"] != raw["source_sha256"]):
            raise ValueError(f"Invalid NPC backup replay in {path.name}")
        seed, condition = result["seed"], result["condition"]
        register(seed, condition, result["observed_sha256"])
        for method, arm in result["arms"].items():
            for row in arm["scores"]:
                add(seed, condition, row["budget"], method, row["training_filter"], row["correct"])

    folder = SOTA / "conditional_votes"
    for path in sorted((folder / "results").glob("case_*.json")):
        result = read(path)
        if (result["source_sha256"] != sha(folder / "conditional_votes.py")
                or result["head_graph_sha256"] != GEOMETRY_SHA or result["test_count"] != 10000):
            raise ValueError(f"Invalid selective compression counts in {path.name}")
        seed, condition = result["seed"], result["condition"]
        register(seed, condition, result["observed_sha256"])
        for method, arm in result["arms"].items():
            for row in arm["records"]:
                depth, budget = row["depth"], row["budget"]
                raw = table.get((seed, condition, budget, method, "aum", f"raw{depth}"))
                if raw is not None and row["correct"]["hard"] != raw:
                    raise ValueError("Selective compression uses a different AUM mask")
                names = {f"{head}{depth}": row["correct"][head] for head in ("conditional", "half")}
                add(seed, condition, budget, method, "all", names)

    folder = SOTA / "capped_votes"
    for path in sorted((folder / "results").rglob("case_*.json")):
        result = read(path)
        fresh = path.parent.name == "fresh"
        entry = "confirm_cap.py" if fresh else "capped_votes.py"
        if (result["source_sha256"] != sha(folder / entry)
                or result["head_graph_sha256"] != GEOMETRY_SHA or result["test_count"] != 10000):
            raise ValueError(f"Invalid cap counts in {path.name}")
        seed, condition = result["seed"], result["condition"]
        register(seed, condition, result["observed_sha256"])
        if fresh:
            baseline = read(SOTA / "npc_backup" / "results" / path.name)
            if (result["cap_source_sha256"] != sha(folder / "capped_votes.py")
                    or (seed, condition) != (baseline["seed"], baseline["condition"])
                    or any(arm["order_sha256"] != baseline["arms"][method]["order_sha256"]
                           for method, arm in result["arms"].items())):
                raise ValueError(f"Capped votes did not replay the NPC anchors in {path.name}")
        for method, arm in result["arms"].items():
            for row in arm["records"]:
                add(seed, condition, row["budget"], method, row["training_filter"],
                    {f'cap{row["depth"]}': row["correct"]})

    folder = SOTA / "backup_mlp"
    for path in sorted((folder / "results").glob("*.json")):
        result = read(path)
        if (result["epochs"] != 300 or result["test_count"] != 10000
                or result["source_sha256"] != sha(folder / "backup_mlp.py")):
            raise ValueError(f"Invalid matched MLP result in {path.name}")
        register(result["seed"], result["condition"], result["observed_sha256"])
        for row in result["cases"]:
            for method, scores in row["scores"].items():
                add(result["seed"], result["condition"], row["budget"], method, "mlp",
                    {name: round(100 * scores[name]) for name in ("mlp", "1nn")})

    columns = ["seed", "condition", "budget", "method", "training_filter", "head", "correct"]
    return pd.DataFrame([(*key, value) for key, value in sorted(table.items())], columns=columns)


def match(data, candidate, candidate_head, reference, reference_head):
    keys = ["condition", "budget", "seed"]
    left = data.loc[data.method.eq(candidate) & data["head"].eq(candidate_head), keys + ["correct"]]
    right = data.loc[data.method.eq(reference) & data["head"].eq(reference_head), keys + ["correct"]]
    pairs = left.merge(right, on=keys, validate="one_to_one", suffixes=("_method", "_reference"))
    delta = pairs.correct_method - pairs.correct_reference
    return pairs.assign(reference_head=reference_head, method_pct=pairs.correct_method / 100,
                        reference_pct=pairs.correct_reference / 100, delta_pp=delta / 100,
                        win=delta.gt(0), tie=delta.eq(0)).drop(columns=["correct_method", "correct_reference"])


def best_reference(scores, heads):
    policies = ("maxherding", "uherding", "js", "observed_10", "soft_10",
                "probcover", "npc", "weighted_npc", "maxherding_nas")
    baseline = scores.loc[scores.method.isin(policies) & scores["head"].isin(heads)]
    winners = baseline.groupby(["seed", "condition", "budget"], as_index=False).correct.max()
    return winners.assign(method="envelope", head="best")


def grouped_scores():
    folder = SOTA / "grouped_backup"
    values = []
    for seed in (70, 71, 72):
        result = read(folder / "results" / f"seed_{seed}.json")
        if (result["geometry_sha256"] != GEOMETRY_SHA or len(result["records"]) != 96
                or result["revisions"]["selector"] != sha(folder / "grouped_backup.py")):
            raise ValueError(f"Invalid grouped backup seed {seed}")
        for row in result["records"]:
            values.append((seed, row["condition"], row["budget"], row["arm"], f'raw{row["depth"]}', row["correct"]))
            if row["depth"] == 3:
                values.append((seed, row["condition"], row["budget"], row["arm"], "1nn", row["nn_correct"]))
    return pd.DataFrame(values, columns=["seed", "condition", "budget", "method", "head", "correct"])
