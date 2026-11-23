"""Task metrics for multiclass and multilabel audio experiments."""
from __future__ import annotations

import numpy as np
from ..task import Task
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score)


def _macro_average_precision(targets: np.ndarray, probabilities: np.ndarray) -> float:
    per_class = [average_precision_score(targets[:, index], probabilities[:, index])
                 for index in range(targets.shape[1]) if np.any(targets[:, index] > 0)]
    return float(np.mean(per_class)) if per_class else 0.0


def choose_thresholds(targets: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    for c in range(targets.shape[1]):
        best = (-1.0, 0.5)
        for threshold in np.linspace(0.05, 0.95, 19):
            score = f1_score(targets[:, c], probabilities[:, c] >= threshold, zero_division=0)
            if score > best[0]: best = (float(score), float(threshold))
        thresholds[c] = best[1]
    return thresholds


def evaluate(targets: np.ndarray, probabilities: np.ndarray, objective: dict,
             thresholds: np.ndarray | None = None) -> dict:
    if Task(objective["task"]) == Task.multiclass:
        pred = probabilities.argmax(axis=1)
        onehot = np.eye(probabilities.shape[1], dtype=np.float32)[targets.astype(int)]
        return {"accuracy": float(accuracy_score(targets, pred)),
                "macro_f1": float(f1_score(targets, pred, average="macro", zero_division=0)),
                "F1": float(f1_score(targets, pred, average="macro", zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(targets, pred)),
                "macro_ap": _macro_average_precision(onehot, probabilities),
                "mAP": _macro_average_precision(onehot, probabilities)}
    thresholds = thresholds if thresholds is not None else np.full(targets.shape[1], float(objective.get("threshold", 0.5)))
    pred = probabilities >= thresholds.reshape(1, -1)
    per_class_ap = [float(average_precision_score(targets[:, i], probabilities[:, i])) if targets[:, i].sum() else None
                    for i in range(targets.shape[1])]
    macro_ap = _macro_average_precision(targets, probabilities)
    return {"macro_ap": macro_ap,
            "mAP": macro_ap,
            "micro_f1": float(f1_score(targets, pred, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(targets, pred, average="macro", zero_division=0)),
            "F1": float(f1_score(targets, pred, average="macro", zero_division=0)),
            "per_class_ap": per_class_ap,
            "per_class_support": targets.sum(axis=0).astype(int).tolist(),
            "class_coverage": float((targets.sum(axis=0) > 0).mean())}
