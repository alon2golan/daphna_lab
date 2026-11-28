"""Compare signed label verification to raw LowBudgetAUM ranking on frozen BYOL."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

import active_audit as archive
from collect import POLICIES, collect, write_csv
from run_official import UPSTREAM_COMMIT, UPSTREAM_SOURCE_SHA256, load_upstream

BUDGETS = (400, 800, 1200, 1600)
SEEDS = tuple(range(80, 85))
CONDITIONS = (("clean", 0.0),) + tuple(
    (f"{kind}{percent}", percent / 100) for kind in ("sym", "asym") for percent in (30, 40, 50)
) + archive.HUMAN_CONDITIONS
METHODS = POLICIES + ("signed_peer", "aum_rank")
ARCHIVE_SHA256 = "72df76b08512438890caa902f8df8f81f0c9f61fe6b5e652833e9b401c10337d"
GEOMETRY_SHA256 = "486509610f2947676c2bf5dfc7d3e6950e79805580290f9ca0d54790058e682b"


class DetectorData:
    """Expose raw features and only the annotations purchased in this replay."""

    def __init__(self, features, warm, observed):
        self.features = features
        self.classes = list(range(archive.CLASSES))
        self.noisy_labels = np.full(len(features), -1, dtype=np.int64)
        self.noisy_labels[warm] = observed

    def __len__(self):
        return len(self.features)


def noise_orders(canonical, seed):
    groups = list(json.loads(archive.GROUPS.read_text()).values())
    if (len(groups) != 20 or any(len(group) != 5 for group in groups)
            or sorted(sum(groups, [])) != list(range(archive.CLASSES))):
        raise ValueError("Invalid CIFAR-100 superclass partition")
    successor = np.empty(archive.CLASSES, dtype=np.int64)
    for group in groups:
        successor[group] = np.roll(group, -1)
    observed = {name: archive.corrupt_labels(canonical, name, rate, seed, successor)
                for name, rate in CONDITIONS if name != "cifar100n"}
    observed["cifar100n"] = archive.human_annotations(canonical)
    return observed


def graph_policies(influence, local, observed, eyes, seed, budget, extra):
    scores = influence @ eyes[observed]
    suspicion, alternative = archive.leave_one_out(local, observed, eyes)
    footprints = archive.counterfactual_footprints(influence, scores, observed, alternative)
    single, diverse = archive.rank_audits(suspicion, footprints, len(influence), extra)
    random = np.random.default_rng(100_000 + seed * 10_000 + budget).choice(budget, extra, replace=False)
    utility = archive.signed_peer_utility(influence, scores, observed, alternative, suspicion)
    return {
        "no_audit": np.empty(0, dtype=int),
        "random_audit": random,
        "loo_audit": np.argsort(-suspicion, kind="stable")[:extra],
        "single_footprint": single,
        "diverse_footprint": diverse,
        "fresh_clean": np.empty(0, dtype=int),
        "signed_peer": np.lexsort((np.arange(budget), -suspicion, -utility))[:extra],
    }


def rank_aum(aum_class, config, seed_everything, features, warm, unqueried, observed, seed, extra):
    seed_everything(700_000 + seed * 10_000 + len(warm))
    detector = aum_class(DetectorData(features, warm, observed), warm, unqueried, config)
    scores = np.asarray(detector.scores)
    if scores.shape != (len(warm),) or not np.isfinite(scores).all():
        raise ValueError("LowBudgetAUM scores are nonfinite or misaligned")
    return np.argsort(scores, kind="stable")[:extra]


def replay(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    geometry = args.geometry.resolve()
    source_sha = archive.sha256(Path(archive.__file__))
    if source_sha != ARCHIVE_SHA256 or archive.sha256(geometry) != GEOMETRY_SHA256:
        raise ValueError("Archived signed code or geometry differs from the verified replay")
    for name, expected in archive.BANK_HASHES.items():
        if archive.sha256(archive.BANK / f"{name}.npy") != expected:
            raise ValueError(f"Frozen BYOL bank checksum mismatch for {name}")
    with np.load(geometry, allow_pickle=False) as payload:
        rows, neighbors, test_neighbors = payload["rows"], payload["neighbors"], payload["heldout_neighbors"]
        provenance = json.loads(payload["metadata"].item())
    if provenance["bank_sha256"] != archive.BANK_HASHES or provenance["groups_sha256"] != archive.sha256(archive.GROUPS):
        raise ValueError("BYOL bank or noise-channel provenance mismatch")

    official, _, aum_config, _ = load_upstream(args.upstream)
    from pycls.lnl.AUM import AUM

    cfg = official.cfg
    cfg.merge_from_file(str(aum_config))
    cfg.DATASET.REPRESENTATION_MODEL = "byol"
    local_features = np.load(archive.BANK / "train_features.npy", mmap_mode="r", allow_pickle=False)[rows]
    all_targets = np.load(archive.BANK / "train_targets.npy", allow_pickle=False)
    truth = np.load(archive.BANK / "test_targets.npy", allow_pickle=False)
    local_targets = all_targets[rows]
    test_features = archive.unit_rows(np.load(archive.BANK / "test_features.npy", allow_pickle=False))
    all_observed = {name: labels[rows] for name, labels in noise_orders(all_targets, args.seed).items()}
    del all_targets
    provenance["human_sha256"] = archive.HUMAN_SHA256
    eyes = np.eye(archive.CLASSES, dtype=np.float32)

    size = BUDGETS[-1] + BUDGETS[-1] // 10
    order = archive.coverage_order(neighbors, size, args.seed)
    train_votes, test_votes = archive.path_counts(neighbors, test_neighbors, order, archive.DEPTH)
    nearest = test_features @ archive.unit_rows(local_features[order]).T
    results = []
    for budget in BUDGETS:
        extra = budget // 10
        used = budget + extra
        warm, fresh = order[:budget], order[budget:used]
        unqueried = np.setdiff1d(np.arange(len(rows)), warm, assume_unique=True)
        target = np.ones(len(rows), dtype=bool)
        target[order[:used]] = False
        influence = train_votes[target, :budget].astype(np.float32)
        local = train_votes[warm, :budget]
        test_base, test_full = test_votes[:, :budget].astype(np.float32), test_votes[:, :used].astype(np.float32)
        pool_full = train_votes[target, :used].astype(np.float32)
        nearest_base, nearest_full = nearest[:, :budget].argmax(1), nearest[:, :used].argmax(1)

        for condition, _ in CONDITIONS:
            observed = all_observed[condition][warm]
            policies = graph_policies(influence, local, observed, eyes, args.seed, budget, extra)
            policies["aum_rank"] = rank_aum(AUM, cfg, official.seed_everything, local_features, warm,
                                          unqueried, observed, args.seed, extra)
            print(f"Finished audit ranks: S{args.seed} B{budget} {condition}", flush=True)
            for policy, requests in policies.items():
                # Only after each ordering is fixed do we use the audit oracle.
                labels = observed.copy()
                labels[requests] = local_targets[warm[requests]]
                expand = policy == "fresh_clean"
                if expand:
                    labels = np.r_[labels, local_targets[fresh]]
                onehot = eyes[labels]
                test_predictions = ((test_full if expand else test_base) @ onehot).argmax(1)
                pool_predictions = ((pool_full if expand else influence) @ onehot).argmax(1)
                results.append({
                    "seed": args.seed, "budget": budget, "condition": condition, "policy": policy,
                    "clean_requests": extra if expand else len(requests),
                    "initial_wrong": int((observed != local_targets[warm]).sum()),
                    "corrected": int((labels[:budget] != observed).sum()),
                    "pool_correct": int((pool_predictions == local_targets[target]).sum()),
                    "pool_count": int(target.sum()), "test_count": len(truth),
                    "test_correct": int((test_predictions == truth).sum()),
                    "test_1nn_correct": int((labels[nearest_full if expand else nearest_base] == truth).sum()),
                    "request_rows": rows[fresh if expand else warm[requests]].tolist(),
                })

    result = {
        "rows": results, "acquisition_rows": rows[order].tolist(),
        "source_sha256": archive.sha256(Path(__file__)), "archive_sha256": source_sha,
        "geometry_sha256": GEOMETRY_SHA256, "provenance": provenance, "depth": archive.DEPTH,
        "numpy": np.__version__, "torch": official.torch.__version__, "gpu": official.torch.cuda.get_device_name(),
        "aum_package": importlib.metadata.version("aum"), "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source_sha256": UPSTREAM_SOURCE_SHA256,
    }
    temporary = output.with_suffix(".tmp")
    with temporary.open("x") as stream:
        json.dump(result, stream, sort_keys=True)
    temporary.rename(output)
    print(f"Saved {len(results)} matched replay evaluations to {output}", flush=True)


def export(args):
    directory = args.directory.resolve()
    indexed = collect(directory, CONDITIONS, SEEDS, BUDGETS, METHODS)
    provenance = {}
    for seed in SEEDS:
        run = json.loads((directory / f"seed_{seed}.json").read_text())
        if (run["geometry_sha256"], run["archive_sha256"], run["upstream_source_sha256"]) != (
                GEOMETRY_SHA256, ARCHIVE_SHA256, UPSTREAM_SOURCE_SHA256):
            raise ValueError(f"Source or geometry provenance mismatch in seed {seed}")
        provenance[seed] = {key: value for key, value in run.items() if key not in ("rows", "acquisition_rows")}
    data = [{key: value for key, value in indexed[idx].items() if key != "request_rows"}
            for idx in sorted(indexed)]
    write_csv(directory / "metrics.csv", data)
    with (directory / "provenance.json").open("x") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True)
    print(f"Verified {len(data)} matched policy evaluations", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    run = actions.add_parser("run")
    run.add_argument("seed", type=int, choices=SEEDS)
    for name in ("geometry", "upstream", "output"):
        run.add_argument(name, type=Path)
    run.set_defaults(handler=replay)
    collector = actions.add_parser("collect")
    collector.add_argument("directory", type=Path)
    collector.set_defaults(handler=export)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
