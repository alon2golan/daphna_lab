"""
noise_cleaning.py

Noise-cleaning methods for low-budget noisy-label active learning.

Methods implemented:
- checkpoint label cleaning
- cross-validation label cleaning over frozen features
- temporal EMA label cleaning
- online AUM-style margin cleaning

All methods edit train_data labels in-place.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

import deep_al.pycls.datasets.utils as ds_utils
from deep_al.pycls.al.lowbudget_aum import lowbudget_aum_noisy_mask


def get_current_labels(train_data: Any) -> np.ndarray:
    if hasattr(train_data, "noisy_labels"):
        return np.asarray(train_data.noisy_labels, dtype=np.int64)
    if hasattr(train_data, "targets"):
        return np.asarray(train_data.targets, dtype=np.int64)
    if hasattr(train_data, "dataset") and hasattr(train_data.dataset, "targets"):
        return np.asarray(train_data.dataset.targets, dtype=np.int64)
    raise RuntimeError("Could not read labels from train_data")


def get_clean_labels(train_data: Any) -> Optional[np.ndarray]:
    if hasattr(train_data, "clean_labels"):
        return np.asarray(train_data.clean_labels, dtype=np.int64)
    return None


def set_one_label(train_data: Any, idx: int, label: int) -> None:
    idx = int(idx)
    label = int(label)

    if hasattr(train_data, "noisy_labels"):
        train_data.noisy_labels[idx] = label

    if hasattr(train_data, "targets"):
        if isinstance(train_data.targets, list):
            train_data.targets[idx] = label
        else:
            train_data.targets[idx] = label

    if hasattr(train_data, "is_noisy") and hasattr(train_data, "clean_labels"):
        train_data.is_noisy[idx] = int(train_data.clean_labels[idx]) != label


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def collect_x_y(train_data: Any, indices: np.ndarray, *, normalize_features: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Collect features/items directly, without DataLoader drop_last.

    In from_features mode, train_data[idx][0] should already be the frozen feature vector.
    We still L2-normalize explicitly for KNN/cross-val consistency.
    """
    xs, ys = [], []
    old_no_aug = getattr(train_data, "no_aug", None)
    if old_no_aug is not None:
        train_data.no_aug = True

    try:
        for idx in np.asarray(indices, dtype=int):
            x, y = train_data[int(idx)]
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
            else:
                x = np.asarray(x)
            xs.append(x)
            ys.append(int(y))
    finally:
        if old_no_aug is not None:
            train_data.no_aug = old_no_aug

    x = np.stack(xs, axis=0)
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    if normalize_features:
        x = l2_normalize(x)

    return x.astype(np.float32), np.asarray(ys, dtype=np.int64)


def predict_with_checkpoint(
    train_data: Any,
    lSet: np.ndarray,
    cfg: Any,
    checkpoint_file: str,
    train_al_module: Any,
) -> Dict[str, np.ndarray]:
    import torch

    labels = get_current_labels(train_data)
    model = train_al_module.mh.get_model(cfg, checkpoint_file)
    device = "cuda" if (getattr(cfg, "NUM_GPUS", 0) > 0 and torch.cuda.is_available()) else "cpu"
    model.to(device)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for start in range(0, len(lSet), cfg.TRAIN.BATCH_SIZE):
            batch_indices = np.asarray(lSet[start:start + cfg.TRAIN.BATCH_SIZE], dtype=int)
            samples = [train_data[int(idx)][0] for idx in batch_indices]
            try:
                inputs = torch.stack([x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in samples]).to(device)
            except Exception:
                inputs = torch.tensor(np.stack([np.asarray(x) for x in samples])).to(device)

            probs = torch.nn.functional.softmax(model(inputs), dim=1).detach().cpu().numpy()
            all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)
    preds = probs.argmax(axis=1).astype(np.int64)
    confs = probs.max(axis=1).astype(np.float32)

    return {
        "indices": np.asarray(lSet, dtype=int),
        "probs": probs,
        "preds": preds,
        "confs": confs,
        "current_labels": labels[np.asarray(lSet, dtype=int)],
    }


def predict_with_crossval(
    train_data: Any,
    lSet: np.ndarray,
    *,
    num_classes: int,
    folds: int = 5,
    max_iter: int = 300,
    normalize_features: bool = True,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Out-of-sample predictions from K-fold linear classifiers over frozen features.

    This is intentionally fast and feature-only. It approximates the current linear head,
    but avoids judging a sample by a model trained directly on that sample.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    indices = np.asarray(lSet, dtype=int)
    x, y = collect_x_y(train_data, indices, normalize_features=normalize_features)

    n = len(indices)
    folds = int(max(2, min(folds, n)))
    probs = np.zeros((n, int(num_classes)), dtype=np.float32)

    splitter = KFold(n_splits=folds, shuffle=True, random_state=int(seed))
    for train_pos, holdout_pos in splitter.split(x):
        y_train = y[train_pos]
        if len(np.unique(y_train)) < 2:
            continue

        clf = LogisticRegression(
            max_iter=int(max_iter),
            solver="lbfgs",
            multi_class="auto",
            n_jobs=-1,
        )
        clf.fit(x[train_pos], y_train)

        fold_probs = clf.predict_proba(x[holdout_pos])
        fold_classes = clf.classes_.astype(int)
        probs[np.asarray(holdout_pos)[:, None], fold_classes[None, :]] = fold_probs.astype(np.float32)

    # If a fold failed to assign probabilities, fallback to one-hot current label.
    missing = probs.sum(axis=1) <= 0
    if np.any(missing):
        probs[missing, y[missing]] = 1.0

    preds = probs.argmax(axis=1).astype(np.int64)
    confs = probs.max(axis=1).astype(np.float32)

    return {
        "indices": indices,
        "probs": probs,
        "preds": preds,
        "confs": confs,
        "current_labels": y,
    }


def _init_cleaning_stats(train_data: Any, indices: np.ndarray) -> Dict[str, Any]:
    labels = get_current_labels(train_data)
    clean_labels = get_clean_labels(train_data)

    stats = {
        "total_noisy_in_lSet": 0,
        "fixed": 0,
        "correctly_fixed": 0,
        "clean_corrupted": 0,
        "wrong_to_wrong": 0,
        "changed_indices": [],
    }

    if clean_labels is not None:
        idx = np.asarray(indices, dtype=int)
        stats["total_noisy_in_lSet"] = int(np.sum(labels[idx] != clean_labels[idx]))

    return stats


def apply_relabels_from_predictions(
    train_data: Any,
    pred_pack: Dict[str, np.ndarray],
    *,
    threshold: float,
    method_name: str,
    extra_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Relabel samples where pred != current_label and confidence >= threshold."""
    indices = pred_pack["indices"].astype(int)
    preds = pred_pack["preds"].astype(int)
    confs = pred_pack["confs"].astype(float)

    labels = get_current_labels(train_data)
    clean_labels = get_clean_labels(train_data)
    stats = _init_cleaning_stats(train_data, indices)

    if extra_mask is None:
        extra_mask = np.ones(len(indices), dtype=bool)
    else:
        extra_mask = np.asarray(extra_mask, dtype=bool)

    for pos, idx in enumerate(indices):
        pred = int(preds[pos])
        conf = float(confs[pos])
        cur_label = int(labels[idx])

        if pred == cur_label or conf < threshold or not bool(extra_mask[pos]):
            continue

        if clean_labels is not None:
            true_label = int(clean_labels[idx])
            if cur_label != true_label and pred == true_label:
                stats["correctly_fixed"] += 1
            elif cur_label == true_label and pred != true_label:
                stats["clean_corrupted"] += 1
            elif cur_label != true_label and pred != true_label:
                stats["wrong_to_wrong"] += 1

        set_one_label(train_data, idx, pred)
        labels[idx] = pred
        stats["fixed"] += 1
        stats["changed_indices"].append(int(idx))

    stats.update({
        "status": "completed",
        "method": method_name,
        "threshold": float(threshold),
    })
    return stats


def run_checkpoint_cleaning(
    train_data: Any,
    lSet: np.ndarray,
    cfg: Any,
    checkpoint_file: Optional[str],
    train_al_module: Any,
    *,
    threshold: float,
) -> Dict[str, Any]:
    if checkpoint_file is None:
        return {"status": "skipped", "reason": "no checkpoint", "method": "label_cleaning"}

    pred_pack = predict_with_checkpoint(train_data, lSet, cfg, checkpoint_file, train_al_module)
    return apply_relabels_from_predictions(
        train_data,
        pred_pack,
        threshold=threshold,
        method_name="label_cleaning",
    )


def run_crossval_cleaning(
    train_data: Any,
    lSet: np.ndarray,
    cfg: Any,
    *,
    threshold: float,
    folds: int,
    max_iter: int,
) -> Dict[str, Any]:
    pred_pack = predict_with_crossval(
        train_data,
        lSet,
        num_classes=int(cfg.MODEL.NUM_CLASSES),
        folds=folds,
        max_iter=max_iter,
        normalize_features=True,
        seed=int(cfg.RNG_SEED or 0),
    )
    result = apply_relabels_from_predictions(
        train_data,
        pred_pack,
        threshold=threshold,
        method_name="crossval_label_cleaning",
    )
    result["folds"] = int(folds)
    result["crossval_max_iter"] = int(max_iter)
    return result


def run_temporal_cleaning(
    train_data: Any,
    lSet: np.ndarray,
    cfg: Any,
    checkpoint_file: Optional[str],
    train_al_module: Any,
    state: Dict[str, Any],
    *,
    threshold: float,
    momentum: float,
    min_seen: int,
) -> Dict[str, Any]:
    if checkpoint_file is None:
        return {"status": "skipped", "reason": "no checkpoint", "method": "temporal"}

    pred_pack = predict_with_checkpoint(train_data, lSet, cfg, checkpoint_file, train_al_module)

    ema_state = state.setdefault("temporal_ema_probs", {})
    count_state = state.setdefault("temporal_seen_count", {})

    ema_probs = []
    for idx, probs in zip(pred_pack["indices"].astype(int), pred_pack["probs"]):
        key = str(int(idx))
        if key in ema_state:
            old = np.asarray(ema_state[key], dtype=np.float32)
            ema = float(momentum) * old + (1.0 - float(momentum)) * probs
        else:
            ema = probs.astype(np.float32)

        ema_state[key] = ema
        count_state[key] = int(count_state.get(key, 0)) + 1
        ema_probs.append(ema)

    ema_probs = np.stack(ema_probs, axis=0)
    pred_pack["probs"] = ema_probs
    pred_pack["preds"] = ema_probs.argmax(axis=1).astype(np.int64)
    pred_pack["confs"] = ema_probs.max(axis=1).astype(np.float32)

    seen = np.asarray([count_state[str(int(idx))] for idx in pred_pack["indices"]], dtype=int)
    stable_mask = seen >= int(min_seen)

    result = apply_relabels_from_predictions(
        train_data,
        pred_pack,
        threshold=threshold,
        method_name="temporal",
        extra_mask=stable_mask,
    )
    result["momentum"] = float(momentum)
    result["min_seen"] = int(min_seen)
    result["eligible_seen_count"] = int(stable_mask.sum())
    return result


def run_aum_cleaning(
    train_data: Any,
    lSet: np.ndarray,
    uSet: np.ndarray,
    cfg: Any,
    checkpoint_file: Optional[str] = None,
    train_al_module: Any = None,
    state: Optional[Dict[str, Any]] = None,
    **_ignored,
) -> Dict[str, Any]:
    """LowBudgetAUM filtering. Does not relabel; returns clean/noisy indices."""
    lSet = np.asarray(lSet, dtype=int)
    uSet = np.asarray(uSet, dtype=int)
    if len(lSet) == 0:
        return {"status": "skipped", "method": "lowbudget_aum", "reason": "empty_labeled_set", "clean_indices": [], "noisy_indices": []}
    labels = get_current_labels(train_data)
    all_features = ds_utils.load_features(
        cfg.DATASET.NAME, train=True, features_tag=cfg.DATASET.get("FEATURES_TAG", None)
    )
    noisy_mask, stats = lowbudget_aum_noisy_mask(
        all_features, labels, lSet, uSet,
        num_classes=int(cfg.MODEL.NUM_CLASSES),
        seed=int(getattr(cfg, "RNG_SEED", 0) or 0),
        epochs=int(getattr(cfg.ACTIVE_LEARNING, "NAS_AUM_EPOCHS", 40)),
        fake_ratio=float(getattr(cfg.ACTIVE_LEARNING, "NAS_AUM_FAKE_RATIO", 1.0)),
        fake_percentile=float(getattr(cfg.ACTIVE_LEARNING, "NAS_AUM_FAKE_PERCENTILE", 80.0)),
        lr=float(getattr(cfg.ACTIVE_LEARNING, "NAS_AUM_LR", 0.1)),
        weight_decay=float(getattr(cfg.ACTIVE_LEARNING, "NAS_AUM_WEIGHT_DECAY", 0.0)),
    )
    clean_indices = lSet[~noisy_mask]
    noisy_indices = lSet[noisy_mask]
    result = {
        "status": "completed", "method": "lowbudget_aum",
        "clean_indices": clean_indices.astype(int).tolist(),
        "noisy_indices": noisy_indices.astype(int).tolist(),
        "pred_clean": int(len(clean_indices)), "pred_noisy": int(len(noisy_indices)),
        **stats,
    }
    clean_labels = get_clean_labels(train_data)
    if clean_labels is not None and len(lSet):
        true_noisy = labels[lSet] != clean_labels[lSet]
        tp = int(np.sum(noisy_mask & true_noisy)); fp = int(np.sum(noisy_mask & (~true_noisy)))
        fn = int(np.sum((~noisy_mask) & true_noisy))
        precision = float(tp / max(1, tp + fp)); recall = float(tp / max(1, tp + fn))
        result.update({
            "oracle_true_noisy": int(true_noisy.sum()), "oracle_correctly_flagged": tp,
            "oracle_clean_flagged": fp, "oracle_missed_noisy": fn,
            "oracle_precision": precision, "oracle_recall": recall,
            "oracle_f1": float(2.0 * precision * recall / max(1e-12, precision + recall)),
        })
    return result
