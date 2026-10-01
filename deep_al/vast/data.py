"""Canonical feature manifests; deliberately bypass legacy image wrappers."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


# Table 1 VAST mean test accuracy (%), keyed by labels per class.
# Reporting references only: never used for training or model selection.
DATASETS = {
    "cifar100": {"classes": 100, "batch_size": 200, "weight": 64.0,
                 "reference": {1: 39.2, 2: 47.8, 4: 54.4}},
    "imagenet100": {"classes": 100, "batch_size": 200, "weight": 64.0,
                    "reference": {1: 34.9, 2: 43.1, 4: 49.3}},
    "tinyimagenet": {"classes": 200, "batch_size": 100, "weight": 32.0,
                     "reference": {1: 17.71, 2: 22.83, 4: 27.55}},
}


def read_config(path):
    with Path(path).open() as stream:
        result = yaml.safe_load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class FeatureData:
    dataset: str
    representation: dict
    class_ids: list
    train_features: np.ndarray
    train_labels: np.ndarray
    train_ids: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    test_ids: np.ndarray
    provenance: dict

    @property
    def classes(self):
        return len(self.class_ids)


def load_manifest(path, labels_per_class=1):
    """Load aligned arrays. The labels encode indices in ordered class_ids."""
    path = Path(path).resolve()
    spec = read_config(path)
    dataset = spec.get("dataset")
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {list(DATASETS)}")
    representation = spec.get("representation", {})
    if not isinstance(representation, dict) or not representation.get("name") or not representation.get("source"):
        raise ValueError("representation needs name and source")
    if not isinstance(spec.get("alignment"), str) or not spec["alignment"].strip():
        raise ValueError("alignment must explain how feature/label/ID ordering was verified")
    hashes, paths = {}, {}

    def resolve(value, key):
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a file path")
        target = (path.parent / value).resolve()
        if not target.is_file():
            raise ValueError(f"Missing {key}: {target}")
        paths[key] = str(target)
        hashes[key] = file_hash(target)
        return target

    class_path = resolve(spec.get("class_ids"), "class_ids")
    class_ids = json.loads(class_path.read_text())
    if (not isinstance(class_ids, list) or not all(isinstance(c, str) and c for c in class_ids)
            or len(set(class_ids)) != len(class_ids)
            or len(class_ids) != DATASETS[dataset]["classes"]):
        raise ValueError("class_ids must be a unique ordered JSON string list of the dataset's classes")
    arrays = {}
    for split in ("train", "test"):
        for kind in ("features", "labels", "sample_ids"):
            key = f"{split}.{kind}"
            target = resolve(spec.get(split, {}).get(kind), key)
            arrays[split, kind] = np.load(target, mmap_mode="r", allow_pickle=False)
        x, y, ids = (arrays[split, k] for k in ("features", "labels", "sample_ids"))
        if (x.ndim != 2 or not len(x) or x.shape[1] == 0 or x.dtype.kind != "f"
                or y.shape != (len(x),) or ids.shape != (len(x),)):
            raise ValueError(f"{split}: expected float features [N,D], labels [N], IDs [N]")
        for start in range(0, len(x), 8192):
            block = np.asarray(x[start:start + 8192], dtype=np.float64)
            if not np.isfinite(block).all() or np.any(np.linalg.norm(block, axis=1) == 0):
                raise ValueError(f"{split}: nonfinite or zero-norm features")
            if np.max(np.abs(block)) > np.finfo(np.float32).max:
                raise ValueError(f"{split}: features exceed float32 classifier range")
        if y.dtype.kind not in "iu" or np.any(y < 0) or np.any(y >= len(class_ids)):
            raise ValueError(f"{split}: labels must be integer indices into class_ids")
        if ids.dtype.kind not in "US" or len(set(ids.tolist())) != len(ids) or np.any(ids.astype(str) == ""):
            raise ValueError(f"{split}: sample IDs must be unique nonempty strings")
    if arrays["train", "features"].shape[1] != arrays["test", "features"].shape[1]:
        raise ValueError("Train/test feature widths differ")
    if set(arrays["train", "sample_ids"].astype(str)) & set(arrays["test", "sample_ids"].astype(str)):
        raise ValueError("Train and test sample IDs overlap")
    counts = np.bincount(arrays["train", "labels"].astype(int), minlength=len(class_ids))
    if np.any(counts < labels_per_class):
        raise ValueError("Insufficient clean training examples for balanced label budget")
    return FeatureData(
        dataset, representation, class_ids,
        arrays["train", "features"], arrays["train", "labels"], arrays["train", "sample_ids"],
        arrays["test", "features"], arrays["test", "labels"], arrays["test", "sample_ids"],
        {"manifest": str(path), "manifest_sha256": file_hash(path), "files": paths,
         "sha256": hashes, "alignment": spec["alignment"], "class_ids": class_ids})


def select_labeled(labels, classes, per_class, seed):
    """Adapted from Itai's random_1c: one private RNG, r draws per class."""
    rng = np.random.RandomState(seed)
    selected = [rng.choice(np.flatnonzero(labels == c), per_class, replace=False)
                for c in range(classes)]
    return np.sort(np.concatenate(selected)).astype(np.int64)
