"""Config-driven audio active-learning experiment endpoint."""
from __future__ import annotations

import hashlib
import csv
import json
import logging
import math
import os
import platform
import random
import tempfile
import time
import importlib.metadata
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..acquisition import AcquisitionContext, Sampler
from ..classifier import fit_head, predict
from ..data.manifest import manifest_digest, read_manifest
from ..data.device import resolve_device
from ..data.compact_utils import cleanup_expanded_batch, extract_audio_batch
from ..extraction.extractor import extract
from ..evaluation.metrics import choose_thresholds, evaluate
from ..evaluation.reporting import write_scores
from ..models.adapters import create_model
from .objectives import encode_targets, labels_from_record, resolve_objective
from ..fine_tune import fit_finetune, predict_records, encode_records
from ..evaluation.dcase_metrics import evaluate_segment_metrics
from ..task import Task
from ..data.dcase import dcase_task_number


def _dcase_task(dataset: str) -> int | None:
    return dcase_task_number(dataset)


def _take_rows(values, indices, device):
    import torch
    rows = torch.as_tensor(indices, dtype=torch.long, device=device)
    return values.index_select(0, rows)


def _to_numpy(values):
    import torch
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _atomic_torch_save(value: Any, path: Path) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temp_name)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _budget_targets(budget: dict[str, Any], pool_size: int) -> list[int]:
    unit = budget.get("unit", "count")
    values = budget.get("values", [])
    if not values:
        raise ValueError("budget.values must contain cumulative budgets")
    if unit == "percent":
        if any(not 0 < float(value) <= 100 for value in values):
            raise ValueError("percentage budgets must be in (0, 100]")
        targets = [int(math.floor(pool_size * float(value) / 100.0 + 0.5)) for value in values]
    elif unit == "count":
        if any(int(value) != float(value) or int(value) <= 0 for value in values):
            raise ValueError("count budgets must be positive integers")
        targets = [int(value) for value in values]
    else:
        raise ValueError("budget.unit must be 'percent' or 'count'")
    if any(target < 1 for target in targets):
        raise ValueError("resolved cumulative query targets must be at least one example")
    if any(b <= a for a, b in zip(targets, targets[1:])):
        raise ValueError("resolved cumulative budget targets must be strictly increasing")
    if any(target > pool_size for target in targets):
        raise ValueError(f"budget target exceeds eligible pool size ({pool_size})")
    return targets


def _make_folds(records: list[dict], dataset: str, dataset_cfg: dict, requested: Any) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    split_names = np.asarray([str(row["split"]).lower() for row in records])
    data = dataset.lower()
    if data == "esc50":
        folds = requested or sorted({int(row["fold"]) for row in records})
        result = []
        for fold in folds:
            test = np.asarray([i for i, row in enumerate(records) if int(row.get("fold", -1)) == int(fold)])
            pool = np.asarray([i for i, row in enumerate(records) if int(row.get("fold", -1)) != int(fold)])
            result.append((f"fold{fold}", pool, np.empty(0, dtype=int), test))
        return result
    if requested and dataset_cfg.get("fold_field", True):
        result = []
        for fold in requested:
            test_mask = np.asarray([str(row.get("fold", "")) == str(fold) and
                                    split_names[i] not in {"evaluation", "eval", "test"}
                                    for i, row in enumerate(records)])
            test = np.flatnonzero(test_mask)
            pool_mask = np.asarray([str(row.get("fold", "")) != str(fold) and
                                    split_names[i] not in {"evaluation", "eval", "test"}
                                    for i, row in enumerate(records)])
            if _dcase_task(data) is not None:
                dev = np.empty(0, dtype=int)
            else:
                dev_mask = np.isin(split_names, ["development", "validation", "dev"])
                pool_mask &= ~dev_mask
                dev = np.flatnonzero(dev_mask)
            pool = np.flatnonzero(pool_mask)
            if len(test) and len(pool): result.append((f"fold{fold}", pool, dev, test))
        if result: return result
    pool_splits = {str(v).lower() for v in dataset_cfg.get("pool_splits", ["train"])}
    val_splits = {str(v).lower() for v in dataset_cfg.get("development_splits", ["development", "validation", "dev"])}
    test_splits = {str(v).lower() for v in dataset_cfg.get("evaluation_splits", ["evaluation", "eval", "test"])}
    pool = np.flatnonzero(np.isin(split_names, list(pool_splits)))
    development = np.flatnonzero(np.isin(split_names, list(val_splits)))
    test = np.flatnonzero(np.isin(split_names, list(test_splits)))
    if not len(pool) or not len(test):
        raise ValueError(f"manifest split mapping produced pool={len(pool)}, test={len(test)}; configure dataset split names explicitly")
    return [("official", pool, development, test)]


def _budget_display(budget: dict, value: float | int) -> str:
    return f"{value:g}{'%' if budget.get('unit') == 'percent' else ' samples'}"


def _validate_provenance(model_cfg: dict) -> dict:
    metadata_path = model_cfg.get("checkpoint_metadata")
    if not metadata_path:
        raise ValueError("SSAST and BEATs runs require model.checkpoint_metadata with self-supervised provenance")
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    required = {"source", "pretraining_data", "license", "evaluation_audio_excluded", "supervised_finetuning"}
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"checkpoint metadata missing fields: {sorted(missing)}")
    if metadata["evaluation_audio_excluded"] is not True or metadata["supervised_finetuning"] is not False:
        raise ValueError("checkpoint must be self-supervised and exclude evaluation audio")
    return metadata


def _validate_feature_cache(cache_dir: Path, manifest: Path, checkpoint: Path,
                            model_name: str, preprocessing: dict, provenance: dict,
                            model_source: Path | None = None) -> dict:
    cache_dir = cache_dir.expanduser().resolve()
    metadata_path = cache_dir / "run.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"feature cache metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_cache_complete") is not True:
        raise ValueError("feature cache is not finalized")
    expected = {"manifest_sha256": manifest_digest(manifest),
                "checkpoint_sha256": _sha256(checkpoint), "model": model_name}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"feature cache conflicts with current run at {key}")
    adapter_path = Path(__file__).resolve().parents[1] / "models" / "adapters.py"
    if metadata.get("adapter_sha256") != _sha256(adapter_path):
        raise ValueError("feature cache was produced with a different model adapter implementation")
    if metadata.get("checkpoint_provenance") != provenance:
        raise ValueError("feature cache checkpoint provenance does not match the current config")
    if model_name == "ssast":
        source = model_source or Path(__file__).resolve().parents[1] / "models_src" / "SSAST" / "src"
        source_file = source / "models" / "ast_models.py"
        if not source_file.is_file() or metadata.get("official_source_sha256") != _sha256(source_file):
            raise ValueError("feature cache was produced with a different SSAST source implementation")
    elif model_name == "beats":
        source = model_source or Path(__file__).resolve().parents[1] / "models_src" / "BEATs"
        digest = hashlib.sha256()
        for name in ("BEATs.py", "modules.py", "backbone.py"):
            source_file = source / name
            if not source_file.is_file():
                raise FileNotFoundError(f"missing BEATs source file: {source_file}")
            digest.update(name.encode())
            digest.update(bytes.fromhex(_sha256(source_file)))
        if metadata.get("official_source_sha256") != digest.hexdigest():
            raise ValueError("feature cache was produced with a different BEATs source implementation")
    effective_preprocessing = {**preprocessing,
                               "window_seconds": float(preprocessing.get("window_seconds", 10.24 if model_name == "ssast" else 10.0)),
                               "channel_mode": preprocessing.get("channel_mode", "mean"),
                               "pad_mode": preprocessing.get("pad_mode", "constant")}
    preprocessing_id = hashlib.sha256(json.dumps(effective_preprocessing, sort_keys=True).encode()).hexdigest()[:12]
    if metadata.get("preprocessing_id") != preprocessing_id:
        raise ValueError("feature cache conflicts with current run at preprocessing_id")
    for name in ("embeddings.npz", "samples.jsonl", "failures.json"):
        if not (cache_dir / name).is_file():
            raise FileNotFoundError(f"feature cache is missing {name}")
        if _sha256(cache_dir / name) != metadata.get("artifact_sha256", {}).get(name):
            raise ValueError(f"feature cache checksum mismatch: {name}")
    expected_ids = {str(row["sample_id"]) for row in read_manifest(manifest)}
    with np.load(cache_dir / "embeddings.npz", allow_pickle=False) as archive:
        sample_ids = [str(value) for value in archive["sample_ids"]]
        embeddings = archive["embeddings"]
    if len(sample_ids) != len(set(sample_ids)) or embeddings.ndim != 2 or len(sample_ids) != len(embeddings):
        raise ValueError("feature cache embedding IDs and matrix are inconsistent")
    if not np.isfinite(embeddings).all():
        raise ValueError("feature cache contains non-finite embeddings")
    failures = json.loads((cache_dir / "failures.json").read_text(encoding="utf-8"))
    failed_ids = [str(item["sample_id"]) for item in failures]
    if len(failed_ids) != len(set(failed_ids)) or set(sample_ids) & set(failed_ids):
        raise ValueError("feature cache has duplicate or conflicting sample failures")
    if set(sample_ids) | set(failed_ids) != expected_ids:
        raise ValueError("feature cache does not account for every manifest sample")
    rows = [json.loads(line) for line in (cache_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if [str(row["sample_id"]) for row in rows] != sample_ids:
        raise ValueError("feature cache sample metadata order does not match embedding IDs")
    return metadata


def _objective_dict(objective) -> dict:
    return asdict(objective)


def _class_vocabulary(dataset_cfg: dict, dataset_name: str, task: str, pool_labels) -> list[str]:
    configured = dataset_cfg.get("classes")
    if configured is not None:
        values = [str(value) for value in configured]
    elif dataset_cfg.get("class_names_file"):
        with Path(dataset_cfg["class_names_file"]).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError("dataset.class_names_file has no rows")
        preferred = "mid" if "mid" in rows[0] else "event_class" if "event_class" in rows[0] else "class_name" if "class_name" in rows[0] else "target"
        if preferred not in rows[0]:
            raise ValueError("class vocabulary CSV must have a mid, event_class, class_name, or target column")
        values = [str(row[preferred]) for row in rows]
    elif Task(task) == Task.multiclass:
        values = sorted({str(label) for label in pool_labels})
    else:
        values = sorted({str(label) for labels in pool_labels for label in labels})
    if len(values) != len(set(values)):
        raise ValueError("class vocabulary contains duplicate identifiers")
    if dataset_name == "as20k" and len(values) != 527:
        raise ValueError("AudioSet-20K requires its complete 527-class vocabulary; set dataset.class_names_file to class_labels_indices.csv")
    return values


def _score_state_path(run_dir: Path, fold_name: str, seed: int, setting_name: str) -> Path:
    safe = setting_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return run_dir / "checkpoints" / f"{fold_name}-seed{seed}-{safe}.json"


def run_experiment(config: dict[str, Any]) -> Path:
    """Run one or more methods, seeds, folds, and configured budget settings."""
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a YAML mapping")
    device = resolve_device(str(config.get("device", "cpu")))
    config = dict(config)
    config["device"] = str(device)
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    al_cfg = config.get("acquisition", {})
    sampler = Sampler()
    budget_cfg = config.get("budget", {})
    train_cfg = config.get("training", {})
    dataset_name = str(dataset_cfg.get("name", "")).lower()
    if dataset_name not in {"esc50", "as20k"} and _dcase_task(dataset_name) is None:
        raise ValueError("dataset.name must be esc50, as20k, or a supported DCASE 2016 task alias")
    mode = str(config.get("adaptation", "frozen_representation")).lower().replace("-", "_")
    if mode not in {"frozen_representation", "linearprob", "fine_tune"}:
        raise ValueError("adaptation must be frozen_representation, linearprob, or fine_tune")
    model_name = str(model_cfg.get("name", "")).lower()
    if model_name not in {"ssast", "beats"}:
        raise ValueError("model.name must be ssast or beats")
    checkpoint = Path(model_cfg["checkpoint"]).expanduser().resolve()
    manifest = Path(dataset_cfg["manifest"]).expanduser().resolve()
    output = Path(config["output"]).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    if output == repo_root or repo_root in output.parents:
        raise ValueError("experiment outputs must be outside the repository")
    if not manifest.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("manifest or checkpoint does not exist")
    provenance = _validate_provenance(model_cfg)
    records = read_manifest(manifest)
    expanded_audio_batches: list[tuple[Path, list[str]]] = []
    expanded_audio_root = None
    objective = _objective_dict(resolve_objective(config.get("objective", {}), dataset_name))
    seeds = [int(s) for s in config.get("seeds", [42, 43, 44, 45, 46])]
    folds = _make_folds(records, dataset_name, dataset_cfg, config.get("folds"))
    all_methods = [str(v).lower().replace("-", "_") for v in (al_cfg.get("methods") or [al_cfg.get("method", "random")])]
    known = {"random", "uncertainty", "least_confidence", "core_set", "kcenter", "k_center", "core-set", "min_margin", "min-margin", "margin", "max_entropy", "max-entropy", "entropy", "bald", "typiclust", "max_herding", "max-herding", "maxherding", "probcover", "cls_al", "cls-al"}
    if set(all_methods) - known:
        raise ValueError(f"unsupported methods: {sorted(set(all_methods) - known)}")
    for fold_name, pool_orig, _, _ in folds:
        _budget_targets(budget_cfg, len(pool_orig))
    if "cls_al" in all_methods and mode != "fine_tune":
        raise ValueError("CLS-AL is a full fine-tuning method; set adaptation: fine_tune")
    if "bald" in all_methods:
        if mode == "linearprob" or float(train_cfg.get("dropout", 0)) <= 0:
            raise ValueError("BALD requires a dropout-enabled task head; linearprob mode has no dropout")
        if int(al_cfg.get("mc_passes", 20)) < 2:
            raise ValueError("BALD requires acquisition.mc_passes >= 2")
    if "probcover" in all_methods:
        deltas = al_cfg.get("deltas", [al_cfg.get("delta", 0.25)])
        for delta in deltas:
            if str(delta).lower() == "auto":
                continue
            if float(delta) <= 0:
                raise ValueError("ProbCover deltas must be positive Euclidean radii or 'auto'")
    if "max_herding" in all_methods and float(al_cfg.get("sigma", al_cfg.get("delta", 0.25))) <= 0:
        raise ValueError("MaxHerding sigma must be positive")
    mode_dir_id = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]
    run_dir = output / f"{dataset_name}-{model_name}-{mode_dir_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=run_dir / "run.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)
    logging.info("starting experiment %s", mode_dir_id)
    feature_cache = config.get("feature_cache")
    if feature_cache:
        extraction_dir = Path(feature_cache).expanduser().resolve()
        _validate_feature_cache(extraction_dir, manifest, checkpoint, model_name,
                                config.get("preprocessing", {}), provenance,
                                Path(model_cfg["source"]).expanduser().resolve() if model_cfg.get("source") else None)
    else:
        extraction_dir = extract(manifest, model_name, checkpoint, run_dir / "feature_cache",
                                 batch_size=int(config.get("extraction", {}).get("batch_size", 16)),
                                 device=str(device),
                                 model_source=Path(model_cfg["source"]).expanduser() if model_cfg.get("source") else None,
                                 checkpoint_metadata=Path(model_cfg["checkpoint_metadata"]).expanduser(),
                                 preprocessing=config.get("preprocessing", {}))
    if mode == "fine_tune" and config.get("audio_index"):
        index_path = Path(config["audio_index"]).expanduser().resolve()
        index_value = json.loads(index_path.read_text(encoding="utf-8"))
        if index_value.get("manifest_sha256") != manifest_digest(manifest):
            raise ValueError("fine-tuning audio index was created for a different manifest")
        temp_parent = Path(config.get("temporary_audio_root", tempfile.gettempdir())).expanduser().resolve()
        temp_parent.mkdir(parents=True, exist_ok=True)
        expanded_audio_root = Path(tempfile.mkdtemp(prefix="audio-al-", dir=temp_parent))
        audio_by_id = {}
        for archive in index_value.get("archives", []):
            batch_root = expanded_audio_root / archive["archive_id"]
            items = extract_audio_batch(index_path, archive["archive_id"], batch_root)
            paths = [item["audio_path"] for item in items]
            expanded_audio_batches.append((batch_root, paths))
            audio_by_id.update({item["sample_id"]: item["audio_path"] for item in items})
        missing = [str(row["sample_id"]) for row in records if str(row["sample_id"]) not in audio_by_id]
        if missing:
            raise ValueError(f"fine-tuning audio index does not contain all manifest samples: {missing[:10]}")
        for row in records:
            row["audio_path"] = audio_by_id[str(row["sample_id"])]
    with np.load(extraction_dir / "embeddings.npz", allow_pickle=False) as archive:
        feature_by_id = {str(i): x for i, x in zip(archive["sample_ids"], archive["embeddings"])}
    rows = json.loads((extraction_dir / "failures.json").read_text(encoding="utf-8")) if (extraction_dir / "failures.json").exists() else []
    failure_count = len(rows)
    failed_ids = {row.get("sample_id") for row in rows}
    missing_audio_count = sum(1 for row in records if row.get("missing_audio") or row.get("sample_id") in failed_ids)
    extraction_metadata = json.loads((extraction_dir / "run.json").read_text(encoding="utf-8"))
    (run_dir / "failures.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    present = [i for i, row in enumerate(records) if row["sample_id"] in feature_by_id]
    index_map = {record_index: matrix_index for matrix_index, record_index in enumerate(present)}
    import torch
    X = torch.as_tensor(
        np.stack([feature_by_id[records[i]["sample_id"]] for i in present]).astype(np.float32),
        dtype=torch.float32,
        device=device,
    )
    feature_transform = str(al_cfg.get("feature_transform", "identity")).lower()
    if feature_transform == "l2":
        acquisition_features = torch.nn.functional.normalize(X, p=2, dim=1, eps=1e-12)
    elif feature_transform in {"identity", "none"}:
        acquisition_features = X
    else:
        raise ValueError("acquisition.feature_transform must be identity or l2")
    mapped_records = [records[i] for i in present]
    raw_scores: list[dict] = []
    scores_path = run_dir / "scores.jsonl"
    if scores_path.exists():
        raw_scores = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    budget_values = budget_cfg.get("values", [])
    if not budget_values: raise ValueError("budget.values is required")
    for fold_name, pool_orig, dev_orig, test_orig in folds:
        pool = np.asarray([index_map[i] for i in pool_orig if i in index_map], dtype=np.int64)
        dev = np.asarray([index_map[i] for i in dev_orig if i in index_map], dtype=np.int64)
        test = np.asarray([index_map[i] for i in test_orig if i in index_map], dtype=np.int64)
        if not len(pool) or not len(test):
            logging.warning("skip %s: pool=%d test=%d embeddings", fold_name, len(pool), len(test)); continue
        targets_for_vocab = [labels_from_record(mapped_records[i], dataset_name, None) for i in pool]
        class_names = _class_vocabulary(dataset_cfg, dataset_name, objective["task"], targets_for_vocab)
        if not class_names: raise ValueError("no classes found among eligible training-pool annotations")
        pool_targets = encode_targets(targets_for_vocab, class_names, type("O", (), objective)())
        dev_targets = encode_targets([labels_from_record(mapped_records[i], dataset_name, None) for i in dev], class_names, type("O", (), objective)()) if len(dev) else None
        test_targets = encode_targets([labels_from_record(mapped_records[i], dataset_name, None) for i in test], class_names, type("O", (), objective)())
        budget_targets = _budget_targets(budget_cfg, len(pool_orig))
        pool_position = {int(global_index): position for position, global_index in enumerate(pool)}
        def target_rows(global_indices):
            return pool_targets[np.asarray([pool_position[int(index)] for index in global_indices], dtype=np.int64)]
        if config.get("report_full_label_reference", True) and not any(
                row.get("method") == "full_label_supervised_reference" and row.get("fold") == fold_name
                for row in raw_scores):
            reference_started = time.perf_counter()
            reference_cfg = dict(train_cfg)
            if mode == "linearprob":
                reference_cfg.update(hidden_sizes=[], dropout=0.0)
            reference_seed = seeds[0] if seeds else 42
            reference_model, _, _ = fit_head(_take_rows(X, pool, device), pool_targets, len(class_names), objective,
                                              reference_cfg, str(device), reference_seed)
            reference_dev = predict(reference_model, _take_rows(X, dev, device), objective, str(device)) if len(dev) else None
            reference_test = predict(reference_model, _take_rows(X, test, device), objective, str(device))
            reference_thresholds = (choose_thresholds(dev_targets, _to_numpy(reference_dev))
                                    if objective["task"] == Task.multilabel and reference_dev is not None
                                    else np.full(len(class_names), float(objective.get("threshold", 0.5) or 0.5)))
            reference_metrics = evaluate(test_targets, _to_numpy(reference_test), objective, reference_thresholds)
            if _dcase_task(dataset_name) == 4:
                reference_metrics.update(evaluate_segment_metrics(mapped_records, test, _to_numpy(reference_test),
                                                                   class_names, reference_thresholds))
            all_pool_targets = pool_targets
            full_positive_counts = (all_pool_targets.sum(axis=0).astype(int).tolist()
                                    if objective["task"] == Task.multilabel else
                                    np.bincount(all_pool_targets, minlength=len(class_names)).astype(int).tolist())
            reference_row = {"dataset": dataset_name, "backbone": model_name,
                             "adaptation": "frozen_representation_full_label_reference",
                             "method": "full_label_supervised_reference", "reference_type": "context_only",
                             "delta": None, "fold": fold_name, "seed": reference_seed,
                             "budget_unit": "count", "nominal_budget": len(pool_orig),
                             "nominal_query_target": len(pool_orig), "actual_queries": len(pool),
                             "missing_audio_count": missing_audio_count,
                             "test_samples_total": len(test_orig), "test_samples_scored": len(test),
                             "preprocessing_id": extraction_metadata.get("preprocessing_id"),
                             "objective": objective,
                             "positive_counts": dict(zip(class_names, full_positive_counts)),
                             "class_coverage": float(np.mean(np.asarray(full_positive_counts) > 0)),
                             "runtime_seconds": time.perf_counter() - reference_started,
                             "thresholds": reference_thresholds.tolist(), "classes": class_names,
                             **reference_metrics}
            raw_scores = [row for row in raw_scores if not all(
                row.get(key) == reference_row.get(key) for key in
                ("dataset", "backbone", "adaptation", "method", "fold", "seed", "nominal_budget"))]
            raw_scores.append(reference_row)
            _atomic_write_text(scores_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in raw_scores))
            write_scores(run_dir, raw_scores)
        for seed in seeds:
            for method in all_methods:
                deltas = (al_cfg.get("deltas", [al_cfg.get("delta", 0.25)]) if method == "probcover"
                          else [al_cfg.get("sigma", al_cfg.get("delta", 0.25))] if method == "max_herding"
                          else [None])
                for delta in deltas:
                    delta_label = str(delta).replace(".", "p")
                    setting = f"{method}-delta{delta_label}" if method in {"probcover", "max_herding"} else method
                    if method == "cls_al" and mode != "fine_tune":
                        raise ValueError("CLS-AL is a full fine-tuning method; set adaptation: fine_tune")
                    state_path = _score_state_path(run_dir, fold_name, seed, setting)
                    state_path.parent.mkdir(parents=True, exist_ok=True)
                    resume_payload = {"manifest_sha256": manifest_digest(manifest),
                                      "checkpoint_sha256": _sha256(checkpoint),
                                      "objective": objective, "config": config,
                                      "seed": seed, "fold": fold_name, "setting": setting}
                    resume_fingerprint = hashlib.sha256(json.dumps(
                        resume_payload, sort_keys=True, default=str).encode()).hexdigest()
                    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
                        "selected": [], "completed": [], "resume_fingerprint": resume_fingerprint}
                    if state.get("resume_fingerprint") != resume_fingerprint:
                        raise ValueError(f"resume state is incompatible with this manifest/checkpoint/task/config/seed: {state_path}")
                    selected = [int(i) for i in state.get("selected", [])]
                    finetune_adapter = create_model(model_name, checkpoint, str(device),
                                                    Path(model_cfg["source"]).expanduser() if model_cfg.get("source") else None) if mode == "fine_tune" else None
                    if finetune_adapter is not None:
                        for parameter in finetune_adapter.model.parameters():
                            parameter.requires_grad_(True)
                    finetune_head = None
                    finetune_shadows = None
                    resume_checkpoint = None
                    finetune_cfg = dict(train_cfg)
                    finetune_cfg.setdefault("learning_rate", 1e-5)
                    finetune_cfg["embedding_dim"] = int(X.shape[1])
                    saved_checkpoint = state.get("model_checkpoint")
                    if mode == "fine_tune" and saved_checkpoint and Path(saved_checkpoint).is_file():
                        import torch
                        from .classifier import build_head
                        from .fine_tune import _clone_shadow
                        saved = torch.load(saved_checkpoint, map_location=device, weights_only=False)
                        resume_checkpoint = saved
                        finetune_adapter.model.load_state_dict(saved["backbone"])
                        finetune_head = build_head(X.shape[1], len(class_names), finetune_cfg, str(device))
                        finetune_head.load_state_dict(saved["head"])
                        if saved.get("shadows"):
                            long_pair = _clone_shadow(finetune_adapter, finetune_head)
                            short_pair = _clone_shadow(finetune_adapter, finetune_head)
                            long_pair[0].model.load_state_dict(saved["shadows"]["long_backbone"])
                            long_pair[1].load_state_dict(saved["shadows"]["long_head"])
                            short_pair[0].model.load_state_dict(saved["shadows"]["short_backbone"])
                            short_pair[1].load_state_dict(saved["shadows"]["short_head"])
                            finetune_shadows = (*long_pair, *short_pair)
                    for budget_index, (user_budget, target_count) in enumerate(zip(budget_values, budget_targets)):
                        if budget_index in state.get("completed", []): continue
                        round_started = time.perf_counter()
                        actual_target = min(target_count, len(pool))
                        additional = actual_target - len(selected)
                        query_metadata = state.get("query_metadata", {})
                        checkpoint_path = state_path.with_suffix(".pt")

                        def checkpoint_callback(phase: str, progress_budget: int, progress_ids: list[int]):
                            progress_ids = [int(value) for value in progress_ids]

                            def persist(payload: dict) -> None:
                                payload.update({"phase": phase, "budget_index": progress_budget,
                                                "selected_ids": progress_ids,
                                                "resume_fingerprint": resume_fingerprint})
                                _atomic_torch_save(payload, checkpoint_path)
                                state["model_checkpoint"] = str(checkpoint_path)
                                state["finetune_progress"] = {"phase": phase,
                                    "budget_index": progress_budget, "selected_ids": progress_ids,
                                    "epoch": payload["epoch"]}
                                _atomic_write_text(state_path, json.dumps(state, indent=2))
                            return persist

                        def resume_args(phase: str, progress_ids: list[int]) -> dict:
                            progress = resume_checkpoint or {}
                            if (progress.get("phase") == phase and
                                    progress.get("budget_index") == budget_index and
                                    progress.get("selected_ids") == [int(value) for value in progress_ids]):
                                return {"optimizer_state": resume_checkpoint.get("optimizer"),
                                        "start_epoch": int(progress.get("epoch", 0)),
                                        "rng_state": resume_checkpoint.get("rng_state")}
                            return {}
                        if additional < 0: raise ValueError("selected query count exceeds configured cumulative target")
                        if additional:
                            candidates = np.asarray([int(i) for i in pool if int(i) not in set(selected)], dtype=np.int64)
                            # Cold-start uncertainty methods with seeded random reveals, counted in the global budget.
                            uncertainty_methods = {"uncertainty", "least_confidence", "min_margin", "margin", "max_entropy", "entropy", "bald", "cls_al"}
                            warm = int(al_cfg.get("initial_random_count", 1)) if method in uncertainty_methods else 0
                            if not selected and warm:
                                n_warm = min(additional, warm)
                                selected.extend(np.random.default_rng(seed).permutation(candidates)[:n_warm].tolist())
                                candidates = np.asarray([i for i in candidates if int(i) not in set(selected)], dtype=np.int64)
                                additional -= n_warm
                            if additional:
                                train_model = long_model = short_model = None
                                probabilities = mc_probabilities = None
                                if method in uncertainty_methods:
                                    if not selected:
                                        raise ValueError(f"{method} requires nonzero initial_random_count")
                                    model_train_cfg = dict(train_cfg)
                                    if mode == "linearprob":
                                        model_train_cfg["hidden_sizes"] = []
                                        model_train_cfg["dropout"] = 0.0
                                    if mode == "fine_tune":
                                        finetune_head, finetune_shadows = fit_finetune(
                                            finetune_adapter, mapped_records, selected, target_rows(selected),
                                            len(class_names), objective, finetune_cfg,
                                            config.get("preprocessing", {}), seed,
                                            head=finetune_head, shadows=finetune_shadows,
                                            checkpoint_callback=checkpoint_callback("acquisition", budget_index, selected),
                                            **resume_args("acquisition", selected))
                                        dynamic = encode_records(finetune_adapter, mapped_records, pool,
                                            config.get("preprocessing", {}), batch_size=int(config.get("extraction", {}).get("batch_size", 16)))
                                        pool_probs = predict(finetune_head, dynamic, objective, str(device))
                                        probabilities = torch.zeros((len(X), len(class_names)), dtype=torch.float32, device=device)
                                        pool_index = torch.as_tensor(pool, dtype=torch.long, device=device)
                                        probabilities[pool_index] = pool_probs
                                        if method == "cls_al":
                                            long_probs = predict(finetune_shadows[1], encode_records(finetune_adapter, mapped_records, pool,
                                                config.get("preprocessing", {}), encoder=finetune_shadows[0].model,
                                                batch_size=int(config.get("extraction", {}).get("batch_size", 16))), objective, str(device))
                                            short_probs = predict(finetune_shadows[3], encode_records(finetune_adapter, mapped_records, pool,
                                                config.get("preprocessing", {}), encoder=finetune_shadows[2].model,
                                                batch_size=int(config.get("extraction", {}).get("batch_size", 16))), objective, str(device))
                                            mc_probabilities = torch.zeros((2, len(X), len(class_names)), dtype=torch.float32, device=device)
                                            mc_probabilities[0, pool_index], mc_probabilities[1, pool_index] = long_probs, short_probs
                                        elif method == "bald":
                                            if float(finetune_cfg.get("dropout", 0)) <= 0:
                                                raise ValueError("BALD requires training.dropout > 0")
                                            mc_pool = predict(finetune_head, dynamic, objective, str(device),
                                                              mc_passes=int(al_cfg.get("mc_passes", 20)))
                                            mc_probabilities = torch.zeros((len(mc_pool), len(X), len(class_names)), dtype=torch.float32, device=device)
                                            mc_probabilities[:, pool_index] = mc_pool
                                    else:
                                        train_model, long_model, short_model = fit_head(_take_rows(X, selected, device), target_rows(selected),
                                              len(class_names), objective, model_train_cfg, str(device), seed,
                                              shadow=(method == "cls_al"))
                                        if method == "cls_al":
                                            probabilities = None
                                            mc_probabilities = torch.zeros((2, len(X), len(class_names)), dtype=torch.float32, device=device)
                                            pool_index = torch.as_tensor(pool, dtype=torch.long, device=device)
                                            mc_probabilities[0, pool_index] = predict(long_model, _take_rows(X, pool, device), objective, str(device))
                                            mc_probabilities[1, pool_index] = predict(short_model, _take_rows(X, pool, device), objective, str(device))
                                        else:
                                            probabilities = torch.zeros((len(X), len(class_names)), dtype=torch.float32, device=device)
                                            pool_index = torch.as_tensor(pool, dtype=torch.long, device=device)
                                            probabilities[pool_index] = predict(train_model, _take_rows(X, pool, device), objective, str(device))
                                        if method == "bald":
                                            if float(model_train_cfg.get("dropout", 0)) <= 0:
                                                raise ValueError("BALD requires training.dropout > 0")
                                            mc_pool = predict(train_model, _take_rows(X, pool, device), objective, str(device),
                                                              mc_passes=int(al_cfg.get("mc_passes", 20)))
                                            mc_probabilities = torch.zeros((len(mc_pool), len(X), len(class_names)), dtype=torch.float32, device=device)
                                            mc_probabilities[:, pool_index] = mc_pool
                                context = AcquisitionContext(
                                    pool_ids=torch.arange(len(X)),
                                    labeled_ids=torch.as_tensor(selected, dtype=torch.long),
                                    candidate_ids=torch.as_tensor(candidates, dtype=torch.long),
                                    features=acquisition_features,
                                    probabilities=probabilities,
                                    mc_probabilities=mc_probabilities,
                                    seed=seed + budget_index,
                                    task=objective["task"],
                                    metadata={"num_classes": len(class_names)},
                                )
                                query_result = sampler.select(
                                    method, context, additional,
                                    delta=(None if method == "probcover" and str(delta).lower() == "auto" else
                                           float(delta) if method == "probcover" else
                                           float(delta) if method == "max_herding" and "sigma" not in al_cfg else None),
                                    sigma=float(delta) if method == "max_herding" else None,
                                )
                                chosen = query_result.selected_ids
                                query_metadata = query_result.metadata
                                selected.extend(chosen.detach().cpu().tolist())
                                state["selected"] = selected
                                state["query_metadata"] = query_metadata
                                _atomic_write_text(state_path, json.dumps(state, indent=2))
                        if not selected:
                            raise RuntimeError("no samples selected for requested budget")
                        fit_cfg = dict(train_cfg)
                        if mode == "linearprob": fit_cfg.update(hidden_sizes=[], dropout=0.0)
                        if mode == "fine_tune":
                            finetune_head, finetune_shadows = fit_finetune(
                                finetune_adapter, mapped_records, selected, target_rows(selected),
                                len(class_names), objective, finetune_cfg, config.get("preprocessing", {}),
                                seed + budget_index, head=finetune_head, shadows=finetune_shadows,
                                checkpoint_callback=checkpoint_callback("evaluation", budget_index, selected),
                                **resume_args("evaluation", selected))
                            dev_probs = predict_records(finetune_adapter, finetune_head, mapped_records, dev,
                                config.get("preprocessing", {}), objective,
                                batch_size=int(config.get("extraction", {}).get("batch_size", 16))) if len(dev) else None
                            test_probs = predict_records(finetune_adapter, finetune_head, mapped_records, test,
                                config.get("preprocessing", {}), objective,
                                batch_size=int(config.get("extraction", {}).get("batch_size", 16)))
                        else:
                            fitted, _, _ = fit_head(_take_rows(X, selected, device), target_rows(selected),
                                                    len(class_names), objective, fit_cfg,
                                str(device), seed + budget_index)
                            dev_probs = predict(fitted, _take_rows(X, dev, device), objective, str(device)) if len(dev) else None
                            test_probs = predict(fitted, _take_rows(X, test, device), objective, str(device))
                        if objective["task"] == Task.multilabel and dev_probs is not None:
                            thresholds = choose_thresholds(dev_targets, _to_numpy(dev_probs))
                        else:
                            thresholds = np.full(len(class_names), float(objective.get("threshold", 0.5) or 0.5))
                        test_probs_np = _to_numpy(test_probs)
                        metric_row = evaluate(test_targets, test_probs_np, objective, thresholds)
                        if _dcase_task(dataset_name) == 4:
                            metric_row.update(evaluate_segment_metrics(mapped_records, test, test_probs_np,
                                                                        class_names, thresholds))
                        selected_targets = target_rows(selected)
                        positive_counts = selected_targets.sum(axis=0).astype(int).tolist() if objective["task"] == Task.multilabel else np.bincount(selected_targets, minlength=len(class_names)).astype(int).tolist()
                        query_row = {"dataset": dataset_name, "backbone": model_name, "adaptation": mode,
                                     "method": method, "delta": (query_metadata.get("delta", delta) if method == "probcover" else delta) if method in {"probcover", "max_herding"} else None,
                                     "delta_selection": query_metadata if method == "probcover" else None,
                                     "sigma": delta if method == "max_herding" else None,
                                     "fold": fold_name, "seed": seed, "budget_unit": budget_cfg.get("unit", "count"),
                                     "nominal_budget": user_budget, "nominal_query_target": target_count,
                                     "actual_queries": len(selected), "budget_shortfall_unavailable_audio": target_count - len(selected),
                                     "missing_audio_count": missing_audio_count,
                                     "test_samples_total": len(test_orig),
                                     "test_samples_scored": len(test),
                                     "preprocessing_id": extraction_metadata.get("preprocessing_id"),
                                     "objective": objective,
                                     "positive_counts": dict(zip(class_names, positive_counts)),
                                     "class_coverage": float(np.mean(np.asarray(positive_counts) > 0)),
                                     "runtime_seconds": time.perf_counter() - round_started,
                                     "thresholds": thresholds.tolist(), "classes": class_names, **metric_row}
                        raw_scores = [r for r in raw_scores if not all(r.get(k) == query_row.get(k) for k in ("dataset", "backbone", "adaptation", "method", "delta", "fold", "seed", "budget_unit", "nominal_budget"))]
                        raw_scores.append(query_row)
                        _atomic_write_text(scores_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in raw_scores))
                        state["completed"] = sorted(set(state.get("completed", []) + [budget_index]))
                        state["selected"] = selected
                        if mode == "fine_tune":
                            import torch
                            saved_state = (torch.load(checkpoint_path, map_location=device, weights_only=False)
                                           if checkpoint_path.is_file() else
                                           {"backbone": finetune_adapter.model.state_dict(),
                                            "head": finetune_head.state_dict(), "shadows": None})
                            saved_state.update({"phase": "completed", "budget_index": budget_index,
                                                "selected_ids": [int(value) for value in selected],
                                                "resume_fingerprint": resume_fingerprint})
                            _atomic_torch_save(saved_state, checkpoint_path)
                            state["model_checkpoint"] = str(checkpoint_path)
                        _atomic_write_text(state_path, json.dumps(state, indent=2))
                        write_scores(run_dir, raw_scores)
                        logging.info("completed %s seed=%d %s budget=%s queries=%d", fold_name, seed, setting, user_budget, len(selected))
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                "manifest": str(manifest), "manifest_sha256": manifest_digest(manifest),
                "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
                "checkpoint_provenance": provenance, "objective": objective, "python": platform.python_version(),
                "preprocessing_id": extraction_metadata.get("preprocessing_id"),
                "missing_audio_count": missing_audio_count,
                "extraction_failure_count": failure_count,
                "cls_al_protocol_difference": "This project uses configurable cumulative global pool query budgets; the paper used 10% initialization and 5% cycles to 40%.",
                "metric_implementation": "scikit-learn average_precision_score/f1_score; sed_eval SegmentBasedMetrics at 1-second resolution for DCASE.",
                "absent_class_ap_policy": "Exclude classes with zero positives from macro AP; retain null per-class AP and zero support.",
                "metric_versions": {name: importlib.metadata.version(name)
                                    for name in ("scikit-learn", "numpy", "sed_eval")},
                "resolved_budget_targets_by_fold": {name: _budget_targets(budget_cfg, len(pool_orig)) for name, pool_orig, _, _ in folds},
                "feature_cache": str(extraction_dir)}
    metadata["acquisition_protocol"] = {
        "methods": all_methods,
        "feature_geometry": feature_transform,
        "geometric_metric": "euclidean (squared distance for greedy ranking)",
        "probcover_radius": al_cfg.get("deltas", [al_cfg.get("delta")]),
        "probcover_comparison": "distance <= delta",
        "max_herding_sigma": al_cfg.get("sigma", al_cfg.get("delta")),
        "max_herding_kernel": "exp(-squared_euclidean / sigma^2)",
        "seed_rule": "experiment seed + cumulative budget index",
        "budget_protocol": "cumulative global pool query targets",
    }
    metadata["metadata_schema_version"] = 2
    run_metadata_path = run_dir / "run.json"
    if run_metadata_path.exists():
        existing = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        # Schema v1 predates explicit acquisition metadata. Migrate those
        # fields in place after validating all fields the old schema knew.
        legacy_additions = {"acquisition_protocol", "metadata_schema_version"}
        for key, value in metadata.items():
            if key != "created_utc" and key not in legacy_additions and existing.get(key) != value:
                raise ValueError(f"Existing experiment metadata conflicts at {key}; refusing to mix runs")
        for key in legacy_additions:
            existing.setdefault(key, metadata[key])
        _atomic_write_text(run_metadata_path, json.dumps(existing, indent=2, default=str))
    else:
        _atomic_write_text(run_metadata_path, json.dumps(metadata, indent=2, default=str))
    write_scores(run_dir, raw_scores)
    for batch_root, paths in expanded_audio_batches:
        cleanup_expanded_batch(batch_root, paths)
    if expanded_audio_root is not None:
        try:
            expanded_audio_root.rmdir()
        except OSError:
            pass
    return run_dir




