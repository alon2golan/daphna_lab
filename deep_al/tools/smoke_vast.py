"""Tiny synthetic end-to-end check; its accuracy is not a scientific result."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deep_al.vast.run import main


def smoke(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args(argv)
    rng = np.random.RandomState(args.seed)
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    centers = rng.normal(size=(100, 8))
    labels = np.repeat(np.arange(100), 4)
    train = centers[labels] + rng.normal(scale=0.01, size=(400, 8))
    spec = {"dataset": "cifar100", "representation": {"name": "synthetic-smoke", "source": "synthetic"},
            "alignment": "Features, labels and unique IDs generated together in row order.",
            "class_ids": "classes.json"}
    (root / "classes.json").write_text(json.dumps([f"synthetic-{i}" for i in range(100)]))
    for split, features, targets in (("train", train, labels), ("test", centers, np.arange(100))):
        spec[split] = {}
        for key, values in (("features", features.astype(np.float32)), ("labels", targets),
                            ("sample_ids", np.array([f"{split}-{i}" for i in range(len(targets))]))):
            filename = f"{split}-{key}.npy"
            np.save(root / filename, values)
            spec[split][key] = filename
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(spec, indent=2))
    print("Synthetic smoke check: 400 train / 100 test examples, 2 epochs; accuracy is not a paper result.", flush=True)
    main(["--manifest", str(manifest), "--labels-per-class", "1", "--seed", str(args.seed),
          "--epochs", "2", "--device", args.device, "--backend", "faiss",
          "--propagation", "literal-paper", "--output-dir", str(root / "run")])


if __name__ == "__main__":
    smoke()
