"""Configurable task objectives and target encoding for audio classifiers."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from ..task import Task
from ..data.dcase import dcase_task_number


@dataclass
class Objective:
    name: str
    task: Task
    output: str
    target: str
    reduction: str
    threshold: float | None = None


def resolve_objective(config: dict, dataset: str) -> Objective:
    multi_default = dataset.lower() in {"as20k", "audioset"} or dcase_task_number(dataset) == 4
    default = {"name": "bce_with_logits", "task": Task.multilabel, "output": "sigmoid", "target": "multi_hot", "reduction": "mean", "threshold": 0.5} if multi_default else {
        "name": "cross_entropy", "task": Task.multiclass, "output": "softmax", "target": "class_index", "reduction": "mean", "threshold": None}
    merged = {**default, **(config or {})}
    try:
        merged["task"] = Task(merged["task"])
    except ValueError as exc:
        raise ValueError("objective.task must be multiclass or multilabel") from exc
    allowed = {"cross_entropy", "bce_with_logits", "focal_bce"}
    if merged["name"] not in allowed:
        raise ValueError(f"unknown objective {merged['name']!r}; registered objectives: {sorted(allowed)}")
    if merged["task"] == Task.multiclass and merged["name"] != "cross_entropy":
        raise ValueError("multiclass objective currently requires cross_entropy")
    if merged["task"] == Task.multilabel and merged["name"] == "cross_entropy":
        raise ValueError("cross_entropy requires multiclass task")
    return Objective(**merged)


def encode_targets(labels: list, class_names: list[str], objective: Objective) -> np.ndarray:
    indices = {label: i for i, label in enumerate(class_names)}
    if objective.task == Task.multiclass:
        result = []
        for label in labels:
            if label not in indices:
                raise ValueError(f"label {label!r} is not in configured class vocabulary")
            result.append(indices[label])
        return np.asarray(result, dtype=np.int64)
    out = np.zeros((len(labels), len(class_names)), dtype=np.float32)
    for row, sample_labels in enumerate(labels):
        for label in sample_labels:
            if label not in indices:
                raise ValueError(f"label {label!r} is absent from configured class vocabulary; provide dataset.classes or dataset.class_names_file")
            out[row, indices[label]] = 1.0
    return out


def labels_from_record(record: dict, dataset: str, objective: Objective) -> list[str] | str:
    labels = record["labels"]
    task_number = dcase_task_number(dataset)
    if task_number == 1:
        return str(labels)
    if task_number == 4:
        return [str(value) for value in labels]
    if dataset.lower() == "esc50":
        return str(labels["target"])
    if dataset.lower() in {"as20k", "audioset"}:
        return [str(value["mid"] if isinstance(value, dict) else value) for value in labels]
    if isinstance(labels, dict) and "target" in labels:
        return str(labels["target"])
    if isinstance(labels, list):
        return [str(value) for value in labels]
    return str(labels)
