# Fine-label -> coarse (superclass) maps, shared by the evaluators (pipeline/) and
# the analysis tools (analysis/). One definition, imported by both.

from pathlib import Path

import numpy as np

# canonical torchvision CIFAR-100 fine(0-99) -> coarse(0-19) superclass map.
# index = fine label id, value = its superclass id. verified 20 groups x 5.
CIFAR100_COARSE = np.array([
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
], dtype=np.int64)


def load_coarse_map(embeddings_dir, dataset="cifar100"):
    # prefer a coarse_map.npy shipped alongside the embeddings (any dataset); else the
    # built-in CIFAR-100 map; else None (the superclass_oracle evaluator will complain).
    p = Path(embeddings_dir) / "coarse_map.npy"
    if p.exists():
        return np.load(p)
    if dataset.lower().startswith("cifar100"):
        return CIFAR100_COARSE
    return None
