"""
noise_utils.py

Simple utilities for noisy-label experiments.

Main convention:
    T[clean_label, noisy_label] = P(noisy_label | clean_label)

Recommended location:
    deep_al/pycls/datasets/noise_utils.py
"""

from pathlib import Path
import json
import pickle

import numpy as np


CIFAR10_ASYM_FLIPS = {
    9: 1,  # truck -> automobile
    2: 0,  # bird -> airplane
    4: 7,  # deer -> horse
    3: 5,  # cat -> dog
    5: 3,  # dog -> cat
}


def validate_transition_matrix(T: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    T = np.asarray(T, dtype=np.float32)

    assert T.ndim == 2
    assert T.shape[0] == T.shape[1]

    if num_classes is not None:
        assert T.shape == (num_classes, num_classes)

    assert np.all(T >= 0)
    assert np.allclose(T.sum(axis=1), 1.0)

    return T


def load_transition_matrix(path: str | Path, num_classes: int | None = None) -> np.ndarray:
    path = Path(path)

    if path.suffix == ".npy":
        T = np.load(path)
    elif path.suffix == ".json":
        obj = json.loads(path.read_text())
        T = obj["transition_matrix"] if isinstance(obj, dict) else obj
    elif path.suffix == ".csv":
        T = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(f"Unsupported transition matrix format: {path.suffix}")

    return validate_transition_matrix(T, num_classes)


def save_transition_matrix(T: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    T = validate_transition_matrix(T)

    if path.suffix == ".npy":
        np.save(path, T)
    elif path.suffix == ".json":
        path.write_text(json.dumps({"transition_matrix": T.tolist()}, indent=2))
    elif path.suffix == ".csv":
        np.savetxt(path, T, delimiter=",")
    else:
        raise ValueError(f"Unsupported transition matrix format: {path.suffix}")


def clean_transition_matrix(num_classes: int) -> np.ndarray:
    return np.eye(num_classes, dtype=np.float32)


def symmetric_transition_matrix(num_classes: int, noise_rate: float) -> np.ndarray:
    T = np.full(
        (num_classes, num_classes),
        noise_rate / (num_classes - 1),
        dtype=np.float32,
    )
    np.fill_diagonal(T, 1.0 - noise_rate)
    return validate_transition_matrix(T, num_classes)


def cifar10_asymmetric_transition_matrix(noise_rate: float) -> np.ndarray:
    T = np.eye(10, dtype=np.float32)

    for clean_label, noisy_label in CIFAR10_ASYM_FLIPS.items():
        T[clean_label, clean_label] = 1.0 - noise_rate
        T[clean_label, noisy_label] = noise_rate

    return validate_transition_matrix(T, 10)


def build_cifar100_superclass_groups_from_torchvision(
    root: str = "./data",
    download: bool = True,
) -> list[list[int]]:
    """Return 20 CIFAR100 superclass groups, each with 5 fine-label indices."""
    from torchvision.datasets import CIFAR100

    dataset = CIFAR100(root=root, train=True, download=download)
    raw_train_path = Path(dataset.root) / dataset.base_folder / dataset.train_list[0][0]

    with open(raw_train_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    fine_labels = np.asarray(raw["fine_labels"])
    coarse_labels = np.asarray(raw["coarse_labels"])

    groups = []
    for coarse_label in sorted(np.unique(coarse_labels)):
        fine_group = sorted(np.unique(fine_labels[coarse_labels == coarse_label]).tolist())
        groups.append(fine_group)

    assert len(groups) == 20
    assert all(len(group) == 5 for group in groups)
    assert sorted(sum(groups, [])) == list(range(100))

    return groups


def save_cifar100_superclass_groups_from_torchvision(
    path: str | Path,
    root: str = "./data",
    download: bool = True,
) -> list[list[int]]:
    """Save readable CIFAR100 superclass groups JSON."""
    from torchvision.datasets import CIFAR100

    groups = build_cifar100_superclass_groups_from_torchvision(root, download)
    dataset = CIFAR100(root=root, train=True, download=False)

    payload = {}
    for i, group in enumerate(groups):
        payload[f"superclass_{i:02d}"] = {
            "fine_label_indices": group,
            "fine_label_names": [dataset.classes[j] for j in group],
        }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))

    return groups


def load_cifar100_superclass_groups(path: str | Path) -> list[list[int]]:
    obj = json.loads(Path(path).read_text())

    groups = []
    for value in obj.values():
        if isinstance(value, dict):
            groups.append(value["fine_label_indices"])
        else:
            groups.append(value)

    groups = [list(map(int, group)) for group in groups]

    assert len(groups) == 20
    assert all(len(group) == 5 for group in groups)
    assert sorted(sum(groups, [])) == list(range(100))

    return groups


def cifar100_asymmetric_transition_matrix(
    noise_rate: float,
    superclass_groups: list[list[int]],
) -> np.ndarray:
    """Flip each CIFAR100 class to the next class inside its superclass."""
    T = np.eye(100, dtype=np.float32)

    for group in superclass_groups:
        for i, clean_label in enumerate(group):
            noisy_label = group[(i + 1) % len(group)]
            T[clean_label, clean_label] = 1.0 - noise_rate
            T[clean_label, noisy_label] = noise_rate

    return validate_transition_matrix(T, 100)


def make_transition_matrix(
    noise_type: str,
    noise_rate: float,
    num_classes: int,
    dataset_name: str,
    transition_path: str | Path | None = None,
    superclass_groups_path: str | Path | None = None,
    data_root: str = "./data",
) -> np.ndarray | None:
    """Create or load a transition matrix.

    Returns None for CIFAR-N human-noise datasets because they use label vectors.
    """
    noise_type = noise_type.lower()
    dataset_name = dataset_name.upper()

    if transition_path is not None:
        return load_transition_matrix(transition_path, num_classes)

    if noise_type in {"clean", "none"}:
        return clean_transition_matrix(num_classes)

    if noise_type in {"sym", "symmetric"}:
        return symmetric_transition_matrix(num_classes, noise_rate)

    if noise_type in {"cifar10n", "cifar100n"}:
        return None

    if noise_type in {"asym", "asymmetric"} and num_classes == 10:
        return cifar10_asymmetric_transition_matrix(noise_rate)

    if noise_type in {"asym", "asymmetric"} and num_classes == 100:
        if superclass_groups_path is None:
            groups = build_cifar100_superclass_groups_from_torchvision(data_root, download=True)
        else:
            groups = load_cifar100_superclass_groups(superclass_groups_path)

        return cifar100_asymmetric_transition_matrix(noise_rate, groups)

    raise ValueError(f"Unsupported noise setup: {noise_type=}, {dataset_name=}, {num_classes=}")


def apply_transition_noise(
    labels: np.ndarray | list[int],
    transition: np.ndarray,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    T = validate_transition_matrix(transition, int(transition.shape[0]))

    rng = np.random.default_rng(seed)
    noisy_labels = np.array([rng.choice(T.shape[0], p=T[y]) for y in labels], dtype=np.int64)
    is_noisy = noisy_labels != labels

    return noisy_labels, is_noisy


def load_cifar_n_labels(pt_path: str | Path, variant: str = "auto") -> np.ndarray:
    import torch

    data = torch.load(pt_path, map_location="cpu")

    if variant != "auto":
        return np.asarray(data[variant], dtype=np.int64)

    # CIFAR-10N usually has aggre_label/worst_label/random_label*.
    # CIFAR-100N official file has noisy_label + clean_label.
    for key in ["aggre_label", "noisy_label", "worst_label", "random_label1", "random_label2", "random_label3"]:
        if key in data:
            return np.asarray(data[key], dtype=np.int64)

    raise KeyError(f"No CIFAR-N noisy-label key found. Available keys: {list(data.keys())}")


def attach_noisy_labels_to_dataset(dataset, noisy_labels: np.ndarray | list[int]):
    clean_labels = np.asarray(dataset.targets, dtype=np.int64)
    noisy_labels = np.asarray(noisy_labels, dtype=np.int64)

    assert len(clean_labels) == len(noisy_labels)

    dataset.clean_labels = clean_labels.copy()
    dataset.noisy_labels = noisy_labels.copy()
    dataset.is_noisy = noisy_labels != clean_labels
    dataset.targets = noisy_labels.tolist()

    return dataset


def noise_summary(clean_labels: np.ndarray | list[int], noisy_labels: np.ndarray | list[int]) -> dict:
    clean_labels = np.asarray(clean_labels, dtype=np.int64)
    noisy_labels = np.asarray(noisy_labels, dtype=np.int64)
    is_noisy = clean_labels != noisy_labels

    return {
        "num_samples": int(len(clean_labels)),
        "num_noisy": int(is_noisy.sum()),
        "noise_rate_realized": float(is_noisy.mean()),
    }
