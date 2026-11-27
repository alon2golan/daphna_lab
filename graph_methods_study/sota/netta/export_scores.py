"""Export verified NPC aggregate scores for the public-image comparison."""

import argparse
from itertools import product
import json
from pathlib import Path

from collect import collect
from netta_baselines import CONDITIONS, METHODS, load_json, sha256


def export(root, output):
    collect(root)
    expected_source = sha256(Path(__file__).with_name("netta_baselines.py"))
    records = []
    for condition, method in product(CONDITIONS, METHODS):
        path = root / condition / f"{method}.json"
        packet = load_json(path)
        if packet["provenance"]["source_sha256"] != expected_source:
            raise ValueError("The matched baseline was run with different code")
        records.append({"condition": condition, "method": method, "artifact_sha256": sha256(path),
                        "provenance": packet["provenance"], "scores": packet["scores"]})
    with output.open("x") as stream:
        json.dump({"source_sha256": expected_source, "records": records}, stream, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.root, args.output)
