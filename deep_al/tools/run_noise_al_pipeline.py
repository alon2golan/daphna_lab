"""
run_noise_al_pipeline.py

Run Active Learning under noisy labels.

Main choices:
- clean validation/test labels are kept clean;
- only the labeled training pool receives noisy labels;
- label-aware AL methods receive the current noisy labels, not clean labels;
- unsupported noise-handling methods fail clearly instead of silently doing nothing.
"""

import argparse
import copy
import itertools
import json
import logging
import os
import sys
import tempfile
import time
import urllib.request
import zipfile
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from deep_al.pycls.al.active_learning import ActiveLearning
from deep_al.pycls.core.config import dump_cfg, initialize_cfg, normalize_cfg_paths
from deep_al.pycls.datasets.data import Data
from deep_al.pycls.datasets.noisy_dataset import NoisyDataset

logger = logging.getLogger("run_noise_al_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

SUPPORTED_NOISE_METHODS = {None, "none", "label_cleaning", "aum"}
CIFAR_N_VARIANTS = {
    "auto",
    "aggre_label",
    "worst_label",
    "clean_label",
    "random_label1",
    "random_label2",
    "random_label3",
}


def _set_new_allowed(cfg_node: Any) -> None:
    if hasattr(cfg_node, "set_new_allowed"):
        cfg_node.set_new_allowed(True)
    if hasattr(cfg_node, "values"):
        for value in cfg_node.values():
            if hasattr(value, "set_new_allowed"):
                _set_new_allowed(value)


def _set_cfg(cfg: Any, key: str, value: Any) -> None:
    if value is None:
        return
    parts = key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = getattr(node, part)
    setattr(node, parts[-1], value)


def _apply_eval_model_alias(args_dict: Dict[str, Any], eval_model_type: Optional[str]) -> None:
    if eval_model_type is None:
        return
    aliases = {
        "NN1": {
            "MODEL.USE_1NN": True,
            "MODEL.LINEAR_FROM_FEATURES": True,
            "MODEL.LINEAR_FROM_IMAGES": False,
        },
        "from_features": {
            "MODEL.USE_1NN": False,
            "MODEL.LINEAR_FROM_FEATURES": True,
            "MODEL.LINEAR_FROM_IMAGES": False,
        },
        "from_images": {
            "MODEL.USE_1NN": False,
            "MODEL.LINEAR_FROM_FEATURES": False,
            "MODEL.LINEAR_FROM_IMAGES": True,
        },
    }
    if eval_model_type not in aliases:
        raise ValueError(f"Unknown eval_model_type={eval_model_type}. Allowed: {sorted(aliases)}")
    args_dict.update(aliases[eval_model_type])


def build_cfg(cfg_file: str, pipeline: Dict[str, Any]) -> Any:
    run_args = pipeline["run_args"]
    args_dict = {
        "RNG_SEED": run_args.get("seed"),
        "EXP_NAME": run_args.get("exp_name"),
        "DATASET.FEATURES_TAG": run_args.get("features_tag"),
        "ACTIVE_LEARNING.SAMPLING_FN": pipeline["active_learning"].get("method"),
        "ACTIVE_LEARNING.BUDGET_SIZE": pipeline["active_learning"].get("budget"),
        "ACTIVE_LEARNING.MAX_ITER": pipeline["active_learning"].get("max_iter"),
        "ACTIVE_LEARNING.INITIAL_DELTA": run_args.get("initial_delta"),
        "ACTIVE_LEARNING.INITIAL_SIGMA": run_args.get("initial_sigma"),
        "KERNEL_TYPE": run_args.get("kernel_type"),
        "EVAL_MODEL_TYPE": run_args.get("eval_model_type"),
        "OPTIM.MAX_EPOCH": run_args.get("max_epoch"),
        "SPARSE_K": run_args.get("sparse_K"),
        "MATRIX_DTYPE": run_args.get("matrix_dtype"),
        "DEBUG": run_args.get("debug"),
        "HIGH_BUDGET": run_args.get("high_budget"),
        "SPARSE_DS": run_args.get("sparse_ds"),
        "ALPHA": run_args.get("alpha"),
        "NORM_IMPORTANCE": run_args.get("norm_importance"),
        "DIFF_METHOD": run_args.get("diff_method"),
        "CONFIDENCE_METHOD": run_args.get("confidence_method"),
        "CONT_METHOD": run_args.get("cont_method"),
        "SOFT_BORDER_VAL": run_args.get("soft_border_val"),
        "DECREASING_ALPHA": run_args.get("decrease_alpha"),
        "ACTIVE_LEARNING.A_LOGISTIC": run_args.get("a_logistic"),
        "ACTIVE_LEARNING.K_LOGISTIC": run_args.get("k_logistic"),
    }
    _apply_eval_model_alias(args_dict, run_args.get("eval_model_type"))

    cfg = initialize_cfg(cfg_file, {k: v for k, v in args_dict.items() if v is not None})
    _set_new_allowed(cfg)
    for k, v in args_dict.items():
        _set_cfg(cfg, k, v)
    normalize_cfg_paths(cfg)
    return cfg


def setup_experiment_directory(cfg: Any) -> str:
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    dataset_out_dir = os.path.join(cfg.OUT_DIR, cfg.DATASET.NAME, cfg.MODEL.TYPE)
    os.makedirs(dataset_out_dir, exist_ok=True)

    now = datetime.now()
    day_dir = os.path.join(dataset_out_dir, f"{now.year}_{now.month}_{now.day}")
    os.makedirs(day_dir, exist_ok=True)

    prefix = "" if cfg.EXP_NAME in [None, "", "auto"] else f"{cfg.EXP_NAME}_"
    sampling_fn = getattr(cfg.ACTIVE_LEARNING, "SAMPLING_FN", "al")
    eval_type = getattr(cfg, "EVAL_MODEL_TYPE", "eval")
    stamp = f"{now.year}_{now.month}_{now.day}_{now.hour:02}{now.minute:02}{now.second:02}_{now.microsecond}"
    exp_name = f"{prefix}{cfg.DATASET.NAME}_{sampling_fn}_{eval_type}_{stamp}"

    cfg.EXP_DIR = os.path.join(day_dir, exp_name)
    os.makedirs(cfg.EXP_DIR, exist_ok=True)
    return cfg.EXP_DIR


def _download_and_extract_cifar_n(save_dir: str) -> Optional[str]:
    os.makedirs(save_dir, exist_ok=True)
    extracted = os.path.join(save_dir, "cifar-10-100n-main")
    if os.path.exists(extracted):
        return extracted

    url = "http://ucsc-real.soe.ucsc.edu:1995/files/cifar-10-100n-main.zip"
    tmp_zip = os.path.join(tempfile.gettempdir(), "cifar-10-100n-main.zip")
    logger.info("Downloading CIFAR-N labels from %s", url)
    try:
        urllib.request.urlretrieve(url, tmp_zip)
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(save_dir)
    except Exception:
        logger.exception("Failed to download/extract CIFAR-N labels")
        return None
    return extracted if os.path.exists(extracted) else save_dir


def _find_cifar_n_pt(extracted_dir: str, noise_type: str) -> Optional[str]:
    import glob

    noise_type = noise_type.lower()
    patterns = ["**/*CIFAR-10*.pt", "**/*CIFAR10*.pt"] if "10" in noise_type else [
        "**/*CIFAR-100*.pt",
        "**/*CIFAR100*.pt",
    ]
    for pattern in patterns:
        matches = glob.glob(os.path.join(extracted_dir, pattern), recursive=True)
        if matches:
            return matches[0]
    return None


def _load_cifar_n_labels(pt_path: str, variant: str) -> np.ndarray:
    import torch

    data = torch.load(pt_path, map_location="cpu")
    if variant != "auto":
        if variant not in data:
            raise KeyError(f"Variant {variant} not found in {pt_path}. Keys: {sorted(data.keys())}")
        return np.asarray(data[variant], dtype=np.int64)

    for key in ["aggre_label", "worst_label", "random_label1", "random_label2", "random_label3"]:
        if key in data:
            return np.asarray(data[key], dtype=np.int64)
    raise KeyError(f"No known CIFAR-N label key found in {pt_path}. Keys: {sorted(data.keys())}")


def _get_targets(dataset: Any) -> Optional[np.ndarray]:
    if hasattr(dataset, "noisy_labels"):
        return np.asarray(dataset.noisy_labels, dtype=np.int64)
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "targets"):
        return np.asarray(dataset.dataset.targets, dtype=np.int64)
    return None


def _set_targets(dataset: Any, labels: np.ndarray) -> None:
    labels = np.asarray(labels, dtype=np.int64)
    if hasattr(dataset, "targets"):
        dataset.targets = labels.tolist() if isinstance(dataset.targets, list) else labels.copy()
    if hasattr(dataset, "noisy_labels"):
        dataset.noisy_labels = labels.copy()


def _set_one_label(dataset: Any, idx: int, label: int) -> None:
    idx = int(idx)
    label = int(label)
    if hasattr(dataset, "noisy_labels"):
        dataset.noisy_labels[idx] = label
    if hasattr(dataset, "targets"):
        if isinstance(dataset.targets, list):
            dataset.targets[idx] = label
        else:
            dataset.targets[idx] = label
    if hasattr(dataset, "is_noisy") and hasattr(dataset, "clean_labels"):
        dataset.is_noisy[idx] = int(dataset.clean_labels[idx]) != label


def _validate_cifar_n_config(cfg: Any, noise_type: str) -> None:
    noise_type = noise_type.lower()
    num_classes = int(cfg.MODEL.NUM_CLASSES)
    if "10" in noise_type and num_classes != 10:
        raise ValueError("cifar10n requires MODEL.NUM_CLASSES=10 and a CIFAR10 config")
    if "100" in noise_type and num_classes != 100:
        raise ValueError("cifar100n requires MODEL.NUM_CLASSES=100 and a CIFAR100 config")


def apply_noise(clean_train_data: Any, cfg: Any, pipeline: Dict[str, Any]) -> tuple[Any, bool]:
    noise_type = pipeline["noise"].get("type")
    noise_rate = float(pipeline["noise"].get("rate") or 0.0)
    if noise_type in [None, "", "clean"] or noise_rate == 0.0 and not str(noise_type).lower().startswith("cifar"):
        return clean_train_data, False

    noise_type = str(noise_type).lower()
    clean_labels = _get_targets(clean_train_data)
    if clean_labels is None:
        raise RuntimeError("Could not read clean labels from train dataset")

    if noise_type.startswith("cifar"):
        _validate_cifar_n_config(cfg, noise_type)
        variant = pipeline["noise"].get("variant", "auto")
        extracted = _download_and_extract_cifar_n(cfg.DATASET.ROOT_DIR)
        if extracted is None:
            raise RuntimeError("Could not download/extract CIFAR-N labels")
        pt_path = _find_cifar_n_pt(extracted, noise_type)
        if pt_path is None:
            raise FileNotFoundError(f"Could not find .pt file for {noise_type} under {extracted}")
        noisy_labels = _load_cifar_n_labels(pt_path, variant)
        if len(noisy_labels) != len(clean_labels):
            raise ValueError(f"CIFAR-N length mismatch: {len(noisy_labels)} labels vs {len(clean_labels)} samples")

        noisy_train_data = copy.copy(clean_train_data)
        noisy_train_data.clean_labels = clean_labels.copy()
        noisy_train_data.noisy_labels = noisy_labels.copy()
        noisy_train_data.is_noisy = noisy_labels != clean_labels
        _set_targets(noisy_train_data, noisy_labels)
        logger.info("Applied %s labels from %s | noisy rate in file: %.4f", noise_type, pt_path, noisy_train_data.is_noisy.mean())
        return noisy_train_data, True

    transition = None
    if pipeline["noise"].get("transition_matrix"):
        transition = np.load(pipeline["noise"]["transition_matrix"])

    noisy_train_data = NoisyDataset(
        copy.copy(clean_train_data),
        noise_type=noise_type,
        noise_rate=noise_rate,
        num_classes=int(cfg.MODEL.NUM_CLASSES),
        seed=cfg.RNG_SEED,
        noise_transition_matrix=transition,
    )
    if not hasattr(noisy_train_data, "clean_labels"):
        noisy_train_data.clean_labels = clean_labels.copy()
    logger.info("Applied synthetic noise: type=%s rate=%.3f", noise_type, noise_rate)
    return noisy_train_data, True


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def get_current_train_labels(train_data: Any) -> np.ndarray:
    labels = _get_targets(train_data)
    if labels is None:
        raise RuntimeError("Could not get labels from noisy train dataset")
    return np.asarray(labels, dtype=np.int64)


def make_loaders(data_obj: Any, cfg: Any, lSet: np.ndarray, valSet: np.ndarray, train_data: Any, clean_val_data: Any, test_data: Any):
    lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=clean_val_data)
    test_loader = data_obj.getTestLoader(data=test_data, test_batch_size=cfg.TRAIN.BATCH_SIZE, seed_id=cfg.RNG_SEED)
    if valSet_loader is None:
        valSet_loader = lSet_loader
    return lSet_loader, valSet_loader, test_loader


def active_sampling_part(cfg, checkpoint_file, cur_episode, data_obj, lSet, train_data, uSet, valSet, clean_val_data, al_obj, train_al_module):
    logger.info("======== ACTIVE SAMPLING ========")
    clf_model = train_al_module.mh.get_model(cfg, checkpoint_file)
    activeSet, new_uSet = al_obj.sample_from_uSet(clf_model, lSet, uSet, train_data, checkpoint_file=checkpoint_file)

    data_obj.saveSets(lSet, uSet, activeSet, cfg.EPISODE_DIR)
    lSet = np.append(lSet, activeSet)
    uSet = new_uSet

    lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=clean_val_data)
    if valSet_loader is None:
        valSet_loader = lSet_loader

    logger.info(
        "After episode %d: labeled=%d unlabeled=%d active=%d",
        cur_episode,
        len(lSet),
        len(uSet),
        len(activeSet),
    )
    return lSet, lSet_loader, uSet, valSet_loader


def run_label_cleaning(train_data: Any, lSet: np.ndarray, cfg: Any, checkpoint_file: Optional[str], train_al_module: Any,
                       threshold: float = 0.9) -> Dict[str, Any]:
    if checkpoint_file is None:
        return {"fixed": 0, "changed_indices": []}

    import torch

    model = train_al_module.mh.get_model(cfg, checkpoint_file)
    device = "cuda" if (cfg.NUM_GPUS > 0 and torch.cuda.is_available()) else "cpu"
    model.to(device)
    model.eval()

    labels = get_current_train_labels(train_data)
    clean_labels = np.asarray(train_data.clean_labels) if hasattr(train_data, "clean_labels") else None

    stats = {
        "total_noisy_in_lSet": 0,
        "correctly_fixed": 0,
        "clean_corrupted": 0,
        "wrong_to_wrong": 0,
    }
    if clean_labels is not None:
        stats["total_noisy_in_lSet"] = int(np.sum(labels[lSet.astype(int)] != clean_labels[lSet.astype(int)]))

    fixed = 0
    changed = []
    with torch.no_grad():
        for start in range(0, len(lSet), cfg.TRAIN.BATCH_SIZE):
            batch_indices = lSet[start:start + cfg.TRAIN.BATCH_SIZE].astype(int)
            samples = [train_data[int(idx)][0] for idx in batch_indices]
            try:
                inputs = torch.stack([x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in samples]).to(device)
            except Exception:
                inputs = torch.tensor(np.stack([np.asarray(x) for x in samples])).to(device)

            probs = torch.nn.functional.softmax(model(inputs), dim=1)
            confs, preds = probs.max(dim=1)
            for idx, pred, conf in zip(batch_indices, preds.cpu().numpy(), confs.cpu().numpy()):
                idx = int(idx)
                pred = int(pred)
                cur_label = int(labels[idx])
                if pred == cur_label or float(conf) < threshold:
                    continue

                if clean_labels is not None:
                    true_label = int(clean_labels[idx])
                    if cur_label != true_label and pred == true_label:
                        stats["correctly_fixed"] += 1
                    elif cur_label == true_label and pred != true_label:
                        stats["clean_corrupted"] += 1
                    elif cur_label != true_label and pred != true_label:
                        stats["wrong_to_wrong"] += 1

                _set_one_label(train_data, idx, pred)
                labels[idx] = pred
                changed.append(idx)
                fixed += 1

    logger.info("Label cleaning changed %d labels | stats=%s", fixed, stats)
    return {"fixed": fixed, "changed_indices": changed, "stats": stats}


def run_aum(train_data: Any, lSet: np.ndarray, cfg: Any, checkpoint_file: Optional[str], train_al_module: Any) -> Dict[str, Any]:
    if checkpoint_file is None:
        return {"status": "skipped", "reason": "no checkpoint"}

    try:
        from aum import AUMCalculator
    except Exception:
        logger.exception("AUM package is unavailable")
        return {"status": "skipped", "reason": "aum package unavailable"}

    import torch

    aum_dir = os.path.join(cfg.EPISODE_DIR, "aum_stats")
    os.makedirs(aum_dir, exist_ok=True)
    aum_calc = AUMCalculator(save_dir=aum_dir)

    labels = get_current_train_labels(train_data)
    model = train_al_module.mh.get_model(cfg, checkpoint_file)
    device = "cuda" if (cfg.NUM_GPUS > 0 and torch.cuda.is_available()) else "cpu"
    model.to(device)
    model.eval()

    with torch.no_grad():
        for start in range(0, len(lSet), cfg.TRAIN.BATCH_SIZE):
            batch_indices = lSet[start:start + cfg.TRAIN.BATCH_SIZE].astype(int)
            samples = [train_data[int(idx)][0] for idx in batch_indices]
            try:
                inputs = torch.stack([x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in samples]).to(device)
            except Exception:
                inputs = torch.tensor(np.stack([np.asarray(x) for x in samples])).to(device)
            logits = model(inputs).cpu().numpy()
            targets = labels[batch_indices]
            aum_calc.update(logits, targets, batch_indices)

    if hasattr(aum_calc, "save"):
        aum_calc.save()
    return {"status": "completed", "aum_dir": aum_dir}


def run_noise_method(method: Optional[str], train_data: Any, lSet: np.ndarray, cfg: Any, checkpoint_file: Optional[str],
                     train_al_module: Any, cur_episode: int, final_val_acc: float, pipeline: Dict[str, Any]) -> Dict[str, Any]:
    method = "none" if method in [None, ""] else str(method).lower()
    if method not in SUPPORTED_NOISE_METHODS:
        raise NotImplementedError(f"noise_method={method} is not implemented. Use: label_cleaning, aum, none")
    if method == "none":
        return {"status": "skipped", "method": "none"}

    warmup_episode = int(pipeline["noise_handling"].get("warmup_episode", 3))
    warmup_min_acc = float(pipeline["noise_handling"].get("warmup_min_acc", 60.0))
    if cur_episode < warmup_episode and final_val_acc < warmup_min_acc:
        return {"status": "skipped", "reason": "warmup"}

    if method == "label_cleaning":
        return run_label_cleaning(train_data, lSet, cfg, checkpoint_file, train_al_module)
    if method == "aum":
        return run_aum(train_data, lSet, cfg, checkpoint_file, train_al_module)
    raise NotImplementedError(method)


def build_pipeline(args: argparse.Namespace, noise_type: Optional[str], al_method: Optional[str], noise_method: Optional[str], exp_name: str) -> Dict[str, Any]:
    if args.cifar_n_variant not in CIFAR_N_VARIANTS:
        raise ValueError(f"Invalid cifar_n_variant={args.cifar_n_variant}")
    noise_method_norm = None if noise_method in [None, ""] else noise_method.lower()
    if noise_method_norm not in SUPPORTED_NOISE_METHODS:
        raise NotImplementedError(f"noise_method={noise_method} is not implemented")

    return {
        "cfg": args.cfg_file,
        "noise": {
            "type": noise_type,
            "rate": args.noise_rate,
            "transition_matrix": args.noise_transition,
            "variant": args.cifar_n_variant,
        },
        "active_learning": {
            "method": al_method,
            "budget": args.budget,
            "max_iter": args.max_iter,
        },
        "noise_handling": {
            "method": noise_method_norm,
            "warmup_episode": args.warmup_episode,
            "warmup_min_acc": args.warmup_min_acc,
        },
        "run_args": {
            "seed": args.seed,
            "initial_size": args.initial_size,
            "exp_name": exp_name,
            "features_tag": args.features_tag,
            "initial_delta": args.initial_delta,
            "initial_sigma": args.initial_sigma,
            "kernel_type": args.kernel_type,
            "eval_model_type": args.eval_model_type,
            "max_epoch": args.max_epoch,
            "sparse_K": args.sparse_K,
            "matrix_dtype": args.matrix_dtype,
            "debug": args.debug,
            "high_budget": args.high_budget,
            "sparse_ds": args.sparse_ds,
            "alpha": args.alpha,
            "norm_importance": args.norm_importance,
            "diff_method": args.diff_method,
            "confidence_method": args.confidence_method,
            "cont_method": args.cont_method,
            "soft_border_val": args.soft_border_val,
            "decrease_alpha": args.decrease_alpha,
            "a_logistic": args.a_logistic,
            "k_logistic": args.k_logistic,
        },
    }


def run_one(pipeline: Dict[str, Any]) -> Dict[str, Any]:
    from deep_al.tools import train_al as train_al_module

    cfg = build_cfg(pipeline["cfg"], pipeline)
    if cfg.RNG_SEED is None:
        cfg.RNG_SEED = int(np.random.randint(100000))
    setup_experiment_directory(cfg)
    dump_cfg(cfg)
    train_al_module.lu.setup_logging(cfg)

    logger.info("Experiment dir: %s", cfg.EXP_DIR)
    data_obj = Data(cfg)
    clean_train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True)
    test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True)

    initial_size = pipeline["run_args"].get("initial_size")
    if initial_size is not None:
        cfg.ACTIVE_LEARNING.INIT_L_RATIO = float(initial_size) / float(train_size)

    lSet_path, uSet_path, valSet_path = data_obj.makeLUVSets(
        train_split_ratio=cfg.ACTIVE_LEARNING.INIT_L_RATIO,
        val_split_ratio=cfg.DATASET.VAL_RATIO,
        data=clean_train_data,
        seed_id=cfg.RNG_SEED,
        save_dir=cfg.EXP_DIR,
    )
    cfg.ACTIVE_LEARNING.LSET_PATH = lSet_path
    cfg.ACTIVE_LEARNING.USET_PATH = uSet_path
    cfg.ACTIVE_LEARNING.VALSET_PATH = valSet_path
    lSet, uSet, valSet = data_obj.loadPartitions(lSetPath=lSet_path, uSetPath=uSet_path, valSetPath=valSet_path)

    train_data, noisy_applied = apply_noise(clean_train_data, cfg, pipeline)
    train_labels = get_current_train_labels(train_data)

    plotting_manager = train_al_module.PlottingManager()
    al_obj = ActiveLearning(data_obj, cfg, train_labels, lSet, plotting_manager)

    lSet_loader, valSet_loader, test_loader = make_loaders(data_obj, cfg, lSet, valSet, train_data, clean_train_data, test_data)

    model = train_al_module.mh.get_model(cfg)
    optimizer = train_al_module.optim.construct_optimizer(cfg, model)
    opt_init_state = deepcopy(optimizer.state_dict()) if not cfg.MODEL.USE_1NN else None
    model_init_state = deepcopy(model.state_dict().copy()) if not cfg.MODEL.USE_1NN else None

    episode_training_times = []
    episode_al_times = []
    noise_method_results = []
    final_train_acc = final_val_acc = final_test_acc = 0.0
    checkpoint_file = None

    for cur_episode in range(0, cfg.ACTIVE_LEARNING.MAX_ITER + 1):
        logger.info("======== EPISODE %d BEGINS ========", cur_episode)
        cfg.EPISODE_DIR = os.path.join(cfg.EXP_DIR, f"episode_{cur_episode}")
        os.makedirs(cfg.EPISODE_DIR, exist_ok=True)

        if cur_episode == 0 and len(lSet) == 0:
            logger.info("Cold start with 0 labels: skipping episode-0 training")
            episode_training_times.append(0.0)
        else:
            try:
                checkpoint_file, train_time, epoch_times, best_train_acc, best_val_acc, test_acc = train_al_module.train_and_eval_model(
                    cfg, cur_episode, lSet_loader, model, optimizer, test_loader, valSet_loader, plotting_manager
                )
            except Exception:
                logger.exception("Training failed")
                return {"status": "failed", "reason": "train_and_eval_model failed", "exp_dir": cfg.EXP_DIR}
            episode_training_times.append(float(train_time))
            final_train_acc = float(best_train_acc)
            final_val_acc = float(best_val_acc)
            final_test_acc = float(test_acc)

        if cur_episode == cfg.ACTIVE_LEARNING.MAX_ITER:
            data_obj.saveSet(lSet, "lSet", cfg.EPISODE_DIR)
            data_obj.saveSet(uSet, "uSet", cfg.EPISODE_DIR)
            break

        start = time.time()
        try:
            lSet, lSet_loader, uSet, valSet_loader = active_sampling_part(
                cfg, checkpoint_file, cur_episode, data_obj, lSet, train_data, uSet, valSet, clean_train_data, al_obj, train_al_module
            )
        except Exception:
            logger.exception("Active sampling failed")
            return {"status": "failed", "reason": "active_sampling_part failed", "exp_dir": cfg.EXP_DIR}
        episode_al_times.append(float(time.time() - start))

        try:
            nm_result = run_noise_method(
                pipeline["noise_handling"].get("method"),
                train_data,
                lSet,
                cfg,
                checkpoint_file,
                train_al_module,
                cur_episode,
                final_val_acc,
                pipeline,
            )
            noise_method_results.append({"episode": cur_episode, "result": nm_result})
        except Exception:
            logger.exception("Noise handling failed")
            return {"status": "failed", "reason": "noise handling failed", "exp_dir": cfg.EXP_DIR}

        if not cfg.ACTIVE_LEARNING.FINE_TUNE:
            model = train_al_module.mh.get_model(cfg, model_init_state=model_init_state)
            optimizer = train_al_module.optim.construct_optimizer(cfg, model, opt_init_state=opt_init_state)

        if checkpoint_file is not None:
            try:
                os.remove(checkpoint_file)
            except OSError:
                pass

    return {
        "status": "completed",
        "exp_dir": cfg.EXP_DIR,
        "summary": {
            "train_size": int(train_size),
            "test_size": int(test_size),
            "final_lSet_len": int(len(lSet)),
            "final_uSet_len": int(len(uSet)),
            "final_valSet_len": int(len(valSet)),
            "noisy_wrapper_applied": bool(noisy_applied),
            "final_train_acc": float(final_train_acc),
            "final_val_acc": float(final_val_acc),
            "final_test_acc": float(final_test_acc),
            "episode_training_times": episode_training_times,
            "episode_al_times": episode_al_times,
            "noise_method_results": noise_method_results,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run noisy-label Active Learning experiments")
    parser.add_argument("--cfg", dest="cfg_file", required=True, type=str)

    parser.add_argument("--noise_type", default=None, type=str)
    parser.add_argument("--noise_rate", default=0.0, type=float)
    parser.add_argument("--noise_transition", default=None, type=str)
    parser.add_argument("--cifar_n_variant", choices=sorted(CIFAR_N_VARIANTS), default="auto", type=str)

    parser.add_argument("--al_method", default=None, type=str)
    parser.add_argument("--budget", default=None, type=int)
    parser.add_argument("--initial_size", default=None, type=int)
    parser.add_argument("--max_iter", default=None, type=int)
    parser.add_argument("--exp_name", default="auto", type=str)

    parser.add_argument("--noise_method", default="none", type=str)
    parser.add_argument("--warmup_episode", default=3, type=int)
    parser.add_argument("--warmup_min_acc", default=60.0, type=float)

    parser.add_argument("--grid_noise_types", default=None, type=str)
    parser.add_argument("--grid_al_methods", default=None, type=str)
    parser.add_argument("--grid_noise_methods", default=None, type=str)
    parser.add_argument("--grid_out_dir", default=None, type=str)

    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--features_tag", default=None, type=str)
    parser.add_argument("--initial_delta", default=None, type=float)
    parser.add_argument("--initial_sigma", default=None, type=float)
    parser.add_argument("--kernel_type", default=None, type=str)
    parser.add_argument("--eval_model_type", default=None, type=str)
    parser.add_argument("--max_epoch", default=None, type=int)

    parser.add_argument("--sparse_K", action="store_true")
    parser.add_argument("--matrix_dtype", choices=["float32", "float16"], default=None, type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--high_budget", action="store_true")
    parser.add_argument("--sparse_ds", action="store_true")
    parser.add_argument("--norm_importance", action="store_true")
    parser.add_argument("--decrease_alpha", action="store_true")
    parser.add_argument("--diff_method", default="abs_diff", type=str)
    parser.add_argument("--confidence_method", default="margin", type=str)
    parser.add_argument("--cont_method", default="positive", type=str)
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument("--a_logistic", default=0.8, type=float)
    parser.add_argument("--k_logistic", default=50, type=int)
    parser.add_argument("--soft_border_val", default=0.5, type=float)

    parser.add_argument("--out", default=None, type=str)
    return parser.parse_args()


def _split_list_arg(value: Optional[str]) -> list[Optional[str]]:
    if value is None:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _safe_name(s: str) -> str:
    keep = []
    for ch in s:
        keep.append(ch if ch.isalnum() or ch in ["-", "_", "."] else "_")
    return "".join(keep)


def main() -> Dict[str, Any]:
    args = parse_args()

    grid_noise = _split_list_arg(args.grid_noise_types) or [args.noise_type]
    grid_al = _split_list_arg(args.grid_al_methods) or [args.al_method]
    grid_noise_methods = _split_list_arg(args.grid_noise_methods) or [args.noise_method]

    grid_out_dir = args.grid_out_dir or os.path.join(os.getcwd(), "grid_runs")
    os.makedirs(grid_out_dir, exist_ok=True)

    all_results = {}
    total = len(grid_noise) * len(grid_al) * len(grid_noise_methods)
    for run_idx, (noise_t, al_m, nmethod) in enumerate(itertools.product(grid_noise, grid_al, grid_noise_methods), start=1):
        combo_name = f"noise={noise_t or 'clean'}__al={al_m or 'cfg'}__nm={nmethod or 'none'}__seed={args.seed or 'NA'}"
        exp_name = args.exp_name if args.exp_name not in [None, "", "auto"] else combo_name
        logger.info("Starting run %d/%d: %s", run_idx, total, combo_name)

        try:
            pipeline = build_pipeline(args, noise_t, al_m, nmethod, exp_name)
            results = run_one(pipeline)
        except Exception as exc:
            logger.exception("Run failed before completion: %s", combo_name)
            pipeline = {"error": "pipeline construction failed"}
            results = {"status": "failed", "reason": repr(exc)}

        out_file = os.path.join(grid_out_dir, f"{_safe_name(combo_name)}.json")
        with open(out_file, "w") as f:
            json.dump(_jsonify({"pipeline": pipeline, "results": results}), f, indent=2)
        all_results[combo_name] = {"pipeline": pipeline, "results": results, "out_file": out_file}

    aggregate_file = os.path.join(grid_out_dir, "aggregate_results.json")
    with open(aggregate_file, "w") as f:
        json.dump(_jsonify(all_results), f, indent=2)
    logger.info("Wrote aggregate results to %s", aggregate_file)
    return all_results


if __name__ == "__main__":
    main()
