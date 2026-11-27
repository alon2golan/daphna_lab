"""Check the runner-up cap on untouched vote weights from fresh NPC acquisitions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import netta_baselines as reference
from npc_backup import CASES
from capped_votes import evaluate


def run(index, archive, output):
    import active_audit as source
    from deep_al.pycls.al import lowbudget_aum
    import torch

    seed, condition = CASES[index]
    expected = reference.load_json(archive / f"case_{index}.json")
    if (expected["seed"], expected["condition"]) != (seed, condition):
        raise ValueError("Wrong fresh baseline packet")
    if expected["detector_sha256"] != reference.DETECTOR_SHA256:
        raise ValueError("Detector provenance changed")
    reference.checked(Path(lowbudget_aum.__file__), reference.DETECTOR_SHA256)
    torch.set_num_threads(2)
    population, nns, test_nns, geometry, rbf, baseline, _, identity = reference.inputs("clean")
    groups = list(json.loads(source.GROUPS.read_text()).values())
    successor = np.empty(reference.CLASSES, dtype=np.int64)
    for group in groups:
        successor[group] = np.roll(group, -1)
    canonical = np.load(source.BANK / "train_targets.npy", allow_pickle=False)
    pool = source.corrupt_labels(canonical, condition, int(condition[-2:]) / 100, seed, successor)
    fingerprint = hashlib.sha256(np.asarray(pool, dtype="<i8").tobytes()).hexdigest()
    if fingerprint != expected["observed_sha256"] or identity["head_graph_sha256"] != expected["head_graph_sha256"]:
        raise ValueError("Fresh oracle or head graph differs")
    features = np.load(source.BANK / "train_features.npy", mmap_mode="r", allow_pickle=False)[population]
    truth = np.load(source.BANK / "test_targets.npy", allow_pickle=False)
    walk, project = (reference.path_matrix(neighbors, len(population)) for neighbors in (nns, test_nns))
    arms = {}
    for method in ("npc", "weighted_npc"):
        order = reference.trajectory(method, features, population, pool, geometry, rbf, baseline,
                                     lowbudget_aum.lowbudget_aum_noisy_mask)
        order_hash = hashlib.sha256(np.asarray(order["rows"], dtype="<i8").tobytes()).hexdigest()
        if order_hash != expected["arms"][method]["order_sha256"]:
            raise ValueError("Fresh acquisition failed exact ordered parity")
        order["masks"] = {str(k): v for k, v in order["masks"].items()}
        arms[method] = {"records": list(evaluate(order, population, walk, project, truth)), "order_sha256": order_hash}
    artifact = {
        "seed": seed, "condition": condition, "test_count": len(truth), "observed_sha256": fingerprint,
        "head_graph_sha256": identity["head_graph_sha256"], "source_sha256": reference.sha256(__file__),
        "cap_source_sha256": reference.sha256(Path(__file__).with_name("capped_votes.py")), "arms": arms,
    }
    with output.open("x") as stream:
        json.dump(artifact, stream, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int, choices=range(8, len(CASES)))
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.index, args.archive, args.output)
