"""Run the unchanged released NPC wrapper on the locked confirmation seeds."""
import argparse
from pathlib import Path

import run_official


SEEDS = (74, 75, 76)
CASES = tuple((condition, seed) for seed in SEEDS for condition in run_official.CONDITIONS)
WRAPPER_SHA = "d5db57a7116111953ffdaa046b9f6369b6f6682f0fd02cbc04511330631db6b4"
GRAPH_SHA = "b76897f7fb38c60d56a9865ca7b3acd8aba7d7aa4af0e9eb4e336b29fda7b892"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=int, choices=range(len(CASES)))
    parser.add_argument("work", type=Path)
    parser.add_argument("official", type=Path)
    args = parser.parse_args()
    condition, seed = CASES[args.task]
    if run_official.file_sha256(Path(run_official.__file__)) != WRAPPER_SHA:
        raise ValueError("Expected the previously verified official NPC wrapper")
    run_official.run(argparse.Namespace(
        upstream=args.official / f"active_learning_with_label_noise-{run_official.UPSTREAM_COMMIT}",
        inputs=args.official / "inputs",
        output=args.work / "runs" / f"{condition}_s{seed}",
        condition=condition,
        seed=seed,
        smoke=False,
    ))
