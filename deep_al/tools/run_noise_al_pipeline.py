"""
run_noise_al_pipeline.py

Generic onboarding script skeleton to run experiments combining:
 - a label-noise type (symmetric/asymmetric/clean/etc.)
 - an active learning method (TypiClust, ProbCover, MISP, etc.)
 - a noise-handling method (label cleaning, robust loss, abstain, etc.)

This script intentionally leaves the three selection arguments empty by default
so they can be filled later during the onboarding experiments.

Inspiration: deep_al/tools/train_al.py, deep_al/tools/train_al_old.py, batch_utils/sweep_train_al.py

Usage (example):
    python run_noise_al_pipeline.py --cfg configs/cifar10.yaml \
        --noise_type sym --noise_rate 0.2 --al_method typiclust --noise_method placeholder

The script builds a simple pipeline and shows where to insert dataset loading,
noisy label injection (NoisyDataset), AL sampling, model training/evaluation and
noise-handling steps.

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
import traceback
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

# Local utility file expected at: deep_al/tools/noise_utils.py
# Keep this import local to avoid accidentally using an older noise_utils under pycls/datasets.
from noise_utils import (
    make_transition_matrix,
    noise_summary,
    save_transition_matrix,
    load_cifar_n_labels,
)
import noise_cleaning as noise_cleaning

logger = logging.getLogger("run_noise_al_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

SUPPORTED_NOISE_METHODS = {None, "none", "label_cleaning", "crossval", "crossval_label_cleaning", "temporal", "aum", "aum_relabel"}
CIFAR_N_VARIANTS = {
    "auto",
    "aggre_label",
    "worst_label",
    "clean_label",
    "random_label1",
    "random_label2",
    "random_label3",
}


# Paper/GitHub defaults used when --paper_hparams is enabled and a value was not
# explicitly supplied from the command line.
COLD_START_AL_METHODS = {
    "typiclust", "typiclust_rp", "typiclust_dc",
    "probcover", "prob_cover", "probcover_orig", "prob_cover_orig",
    "npc_probcover_crossval", "npc_probcover_knn", "npc_probcover_aum", "npc_probcover_oracle",
    "dcom",
    "maxherding", "max_herding", "maxherding_orig", "max_herding_orig",
    "misp", "mispc", "misp_plus", "all_misp",
}

MODEL_DEPENDENT_AL_METHODS = {
    "uncertainty", "entropy", "margin", "coreset", "bald", "dbal", "vaal", "ensemble_var_r", "ensemble_var_R",
}


DCOM_DELTA_TABLE = {
    "STL10": 0.55,
    "SVHN": 0.40,
    "CIFAR10": 0.75,
    "CIFAR100": 0.65,
    "IMAGENET50": 0.75,
    "IMAGENET100": 0.70,
    "IMAGENET200": 0.65,
}

# These are from the MaxHerding/uherding public script. They are useful fallback
# values for datasets that are missing from the DCoM table.
PROBCOVER_CODE_DELTA_TABLE = {
    "CIFAR10": 0.55,
    "CIFAR100": 0.50,
    "STL10": 0.60,
    "TINYIMAGENET": 0.55,
    "IMAGENET": 0.55,
    "IMAGENET100": 0.55,
}

MISP_LOCAL_DEFAULTS = {
    "ALPHA": 0.5,
    "DIFF_METHOD": "abs_diff",
    "CONFIDENCE_METHOD": "margin",
    "CONT_METHOD": "positive",
    "SOFT_BORDER_VAL": 0.5,
}


def _normalize_al_name(name: Optional[str]) -> str:
    return "" if name is None else str(name).lower().replace("-", "_")


def _dataset_key(cfg: Any) -> str:
    name = str(getattr(cfg.DATASET, "NAME", "")).upper()
    name = name.replace("-", "").replace("_", "")
    if "CIFAR100" in name:
        return "CIFAR100"
    if "CIFAR10" in name:
        return "CIFAR10"
    if "TINY" in name and "IMAGENET" in name:
        return "TINYIMAGENET"
    if "IMAGENET50" in name:
        return "IMAGENET50"
    if "IMAGENET100" in name:
        return "IMAGENET100"
    if "IMAGENET200" in name:
        return "IMAGENET200"
    if "IMAGENET" in name:
        return "IMAGENET"
    if "STL10" in name:
        return "STL10"
    if "SVHN" in name:
        return "SVHN"
    return name


def _is_imagenet_family(dataset_key: str) -> bool:
    return dataset_key.startswith("IMAGENET")


def _default_budget(cfg: Any) -> int:
    return int(getattr(cfg.MODEL, "NUM_CLASSES", 10))


def _set_default_cfg(
    cfg: Any,
    pipeline: Dict[str, Any],
    applied: Dict[str, Dict[str, Any]],
    cfg_key: str,
    value: Any,
    reason: str,
    *,
    run_arg: Optional[str] = None,
    active_arg: Optional[str] = None,
    force: bool = False,
) -> None:
    if run_arg is not None and pipeline["run_args"].get(run_arg) is not None and not force:
        return
    if active_arg is not None and pipeline["active_learning"].get(active_arg) is not None and not force:
        return
    _set_cfg(cfg, cfg_key, value)
    if run_arg is not None:
        pipeline["run_args"][run_arg] = value
    if active_arg is not None:
        pipeline["active_learning"][active_arg] = value
    applied[cfg_key] = {"value": value, "reason": reason}


def _set_default_run_arg(
    pipeline: Dict[str, Any],
    applied: Dict[str, Dict[str, Any]],
    run_arg: str,
    value: Any,
    reason: str,
    *,
    force: bool = False,
) -> None:
    if pipeline["run_args"].get(run_arg) is not None and not force:
        return
    pipeline["run_args"][run_arg] = value
    applied[f"run_args.{run_arg}"] = {"value": value, "reason": reason}


def _delta_from_table(cfg: Any, *, prefer_dcom_table: bool = True) -> tuple[Optional[float], str]:
    ds_key = _dataset_key(cfg)
    features_tag = str(getattr(cfg.DATASET, "FEATURES_TAG", "") or "").lower()

    if ds_key == "CIFAR100" and "moco" in features_tag:
        return 0.50, "DCoM appendix: CIFAR-100 MOCOv2+ delta"
    if ds_key == "CIFAR100" and "byol" in features_tag:
        return 0.50, "DCoM appendix: CIFAR-100 BYOL delta"
    if ds_key == "CIFAR100" and "barlow" in features_tag:
        return 0.55, "DCoM appendix: CIFAR-100 Barlow Twins delta"

    if prefer_dcom_table and ds_key in DCOM_DELTA_TABLE:
        return DCOM_DELTA_TABLE[ds_key], "DCoM appendix / ProbCover CIFAR deltas"
    if ds_key in PROBCOVER_CODE_DELTA_TABLE:
        return PROBCOVER_CODE_DELTA_TABLE[ds_key], "uherding GitHub fallback delta"
    return None, "no sourced delta found"


def _apply_training_paper_defaults(cfg: Any, pipeline: Dict[str, Any], applied: Dict[str, Dict[str, Any]], al_name: str) -> None:
    ds_key = _dataset_key(cfg)

    if al_name == "dcom":
        if ds_key == "IMAGENET50":
            max_epoch, base_lr, batch_size = 50, 0.01, 50
        elif ds_key in {"IMAGENET100", "IMAGENET200"}:
            max_epoch, base_lr, batch_size = 100, 0.01, 50
        else:
            max_epoch, base_lr, batch_size = 200, 0.025, 100

        _set_default_cfg(cfg, pipeline, applied, "OPTIM.MAX_EPOCH", max_epoch, "DCoM supervised training", run_arg="max_epoch")
        _set_default_cfg(cfg, pipeline, applied, "OPTIM.BASE_LR", base_lr, "DCoM supervised training", run_arg="base_lr")
        _set_default_cfg(cfg, pipeline, applied, "TRAIN.BATCH_SIZE", batch_size, "DCoM supervised training", run_arg="train_batch_size")
        _set_default_cfg(cfg, pipeline, applied, "TEST.BATCH_SIZE", batch_size, "DCoM supervised training", run_arg="test_batch_size")
        _set_default_cfg(cfg, pipeline, applied, "OPTIM.WEIGHT_DECAY", 3e-4, "DCoM supervised training", run_arg="weight_decay")
        _set_default_cfg(cfg, pipeline, applied, "OPTIM.MOMENTUM", 0.9, "DCoM supervised training", run_arg="momentum")
        _set_default_cfg(cfg, pipeline, applied, "OPTIM.NESTEROV", True, "DCoM supervised training", run_arg="nesterov")
        return

    # TypiClust/ProbCover/Deep-AL config defaults.
    if not _is_imagenet_family(ds_key):
        _set_default_cfg(cfg, pipeline, applied, "OPTIM.MAX_EPOCH", 200, "TypiClust/Deep-AL config default", run_arg="max_epoch")
    _set_default_cfg(cfg, pipeline, applied, "OPTIM.BASE_LR", 0.1, "TypiClust/Deep-AL config default", run_arg="base_lr")
    _set_default_cfg(cfg, pipeline, applied, "OPTIM.WEIGHT_DECAY", 5e-4, "TypiClust/Deep-AL config default", run_arg="weight_decay")
    _set_default_cfg(cfg, pipeline, applied, "OPTIM.MOMENTUM", 0.9, "TypiClust/Deep-AL config default", run_arg="momentum")
    _set_default_cfg(cfg, pipeline, applied, "OPTIM.NESTEROV", False, "TypiClust/Deep-AL config default", run_arg="nesterov")
    _set_default_cfg(cfg, pipeline, applied, "TRAIN.BATCH_SIZE", 128, "TypiClust/Deep-AL config default", run_arg="train_batch_size")


def apply_paper_hparams(cfg: Any, pipeline: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Apply method-specific paper/GitHub defaults unless the user supplied a value."""
    if not pipeline.get("paper_hparams", True):
        pipeline["paper_hparams_applied"] = {}
        return {}

    applied: Dict[str, Dict[str, Any]] = {}
    al_name = _normalize_al_name(pipeline["active_learning"].get("method") or getattr(cfg.ACTIVE_LEARNING, "SAMPLING_FN", None))
    ds_key = _dataset_key(cfg)
    budget = _default_budget(cfg)

    _set_default_cfg(cfg, pipeline, applied, "ACTIVE_LEARNING.BUDGET_SIZE", budget, "paper low-budget setting: budget = number of classes", active_arg="budget")

    if al_name in COLD_START_AL_METHODS or al_name == "random":
        _set_default_run_arg(pipeline, applied, "initial_size", 0, "cold-start setting used by representation-based low-budget AL")
    elif al_name in MODEL_DEPENDENT_AL_METHODS:
        _set_default_run_arg(pipeline, applied, "initial_size", int(pipeline["active_learning"].get("budget") or budget), "model-dependent AL needs an initial labeled pool")

    if al_name in {"maxherding", "max_herding", "maxherding_orig", "max_herding_orig"}:
        _set_default_cfg(cfg, pipeline, applied, "ACTIVE_LEARNING.MAX_ITER", 5 if _is_imagenet_family(ds_key) else 10, "MaxHerding experiments", active_arg="max_iter")
    else:
        _set_default_cfg(cfg, pipeline, applied, "ACTIVE_LEARNING.MAX_ITER", 5, "TypiClust/ProbCover/DCoM default AL rounds", active_arg="max_iter")

    if al_name.startswith("typiclust"):
        # TypiClust constants are class constants in typiclust.py: K_NN=20, MAX_NUM_CLUSTERS=500, MIN_CLUSTER_SIZE=5.
        pass

    elif al_name in {"probcover", "prob_cover", "probcover_orig", "prob_cover_orig", "npc_probcover_crossval", "npc_probcover_knn", "npc_probcover_aum", "npc_probcover_oracle"}:
        _set_default_cfg(cfg, pipeline, applied, "KERNEL_TYPE", "tophat", "ProbCover uses radius balls / TopHat kernel", run_arg="kernel_type")
        _set_default_cfg(cfg, pipeline, applied, "SPARSE_K", True, "ProbCover graph is sparse coverage", run_arg="sparse_K")
        delta, reason = _delta_from_table(cfg, prefer_dcom_table=True)
        if delta is not None:
            _set_default_cfg(cfg, pipeline, applied, "ACTIVE_LEARNING.INITIAL_DELTA", delta, reason, run_arg="initial_delta")

    elif al_name == "dcom":
        delta, reason = _delta_from_table(cfg, prefer_dcom_table=True)
        if delta is not None:
            _set_default_cfg(cfg, pipeline, applied, "ACTIVE_LEARNING.INITIAL_DELTA", delta, reason, run_arg="initial_delta")
        _set_default_cfg(cfg, pipeline, applied, "ACTIVE_LEARNING.MAX_DELTA", 1.1, "DCoM binary-search upper delta", run_arg="max_delta")
        _set_default_cfg(cfg, pipeline, applied, "ACTIVE_LEARNING.DELTA_RESOLUTION", 0.05, "DCoM binary-search resolution", run_arg="delta_resolution")
        _set_default_cfg(cfg, pipeline, applied, "ACTIVE_LEARNING.K_LOGISTIC", 30, "DCoM competence-score logistic parameter", run_arg="k_logistic")
        _set_default_cfg(cfg, pipeline, applied, "ACTIVE_LEARNING.A_LOGISTIC", 0.9 if budget <= 10 else 0.8, "DCoM competence-score logistic parameter", run_arg="a_logistic")

    elif al_name in {"maxherding", "max_herding", "maxherding_orig", "max_herding_orig"}:
        _set_default_cfg(cfg, pipeline, applied, "KERNEL_TYPE", "rbf", "MaxHerding uses Gaussian/RBF kernel", run_arg="kernel_type")
        _set_default_cfg(cfg, pipeline, applied, "ACTIVE_LEARNING.INITIAL_SIGMA", 1.0, "MaxHerding lengthscale for normalized features", run_arg="initial_sigma")
        _set_default_cfg(cfg, pipeline, applied, "ACTIVE_LEARNING.INITIAL_DELTA", 1.0, "MaxHerding original code uses delta/lengthscale=1", run_arg="initial_delta")
        _set_default_cfg(cfg, pipeline, applied, "SPARSE_K", True, "Avoid dense RBF kernel OOM; keep only non-negligible RBF entries", run_arg="sparse_K")
        _set_default_cfg(cfg, pipeline, applied, "MATRIX_DTYPE", "float16", "Reduce kernel memory for MaxHerding", run_arg="matrix_dtype")

    elif al_name in {"bald", "dbal"}:
        _set_default_cfg(cfg, pipeline, applied, "ACTIVE_LEARNING.DROPOUT_ITERATIONS", 25, "Deep-AL/TypiClust config default for MC-dropout methods", run_arg="dropout_iterations")

    elif al_name == "vaal":
        vaal_defaults = {
            "VAAL.Z_DIM": 32,
            "VAAL.VAE_BS": 64,
            "VAAL.VAE_EPOCHS": 100,
            "VAAL.VAE_LR": 5e-4,
            "VAAL.DISC_LR": 5e-4,
            "VAAL.BETA": 1.0,
            "VAAL.ADVERSARY_PARAM": 1.0,
            "VAAL.IM_SIZE": 32,
        }
        for key, value in vaal_defaults.items():
            _set_default_cfg(cfg, pipeline, applied, key, value, "VAAL/Deep-AL config default")

    elif al_name in {"misp", "mispc", "misp_plus", "all_misp"}:
        # I could not find a public paper/GitHub standard for this local method.
        for key, value in MISP_LOCAL_DEFAULTS.items():
            run_arg = {
                "ALPHA": "alpha",
                "DIFF_METHOD": "diff_method",
                "CONFIDENCE_METHOD": "confidence_method",
                "CONT_METHOD": "cont_method",
                "SOFT_BORDER_VAL": "soft_border_val",
            }[key]
            _set_default_cfg(cfg, pipeline, applied, key, value, "local MISP default; no public original HP found", run_arg=run_arg)
        delta, reason = _delta_from_table(cfg, prefer_dcom_table=True)
        if delta is not None:
            _set_default_cfg(cfg, pipeline, applied, "ACTIVE_LEARNING.INITIAL_DELTA", delta, f"MISP local fallback from coverage delta table: {reason}", run_arg="initial_delta")

    _apply_training_paper_defaults(cfg, pipeline, applied, al_name)
    pipeline["paper_hparams_applied"] = applied
    return applied


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
        "MODEL.LOSS_FUN": run_args.get("loss_fun"),
        "MODEL.GCE_Q": run_args.get("gce_q"),
        "MODEL.BOOTSTRAP_BETA": run_args.get("bootstrap_beta"),
        "OPTIM.MAX_EPOCH": run_args.get("max_epoch"),
        "OPTIM.BASE_LR": run_args.get("base_lr"),
        "OPTIM.WEIGHT_DECAY": run_args.get("weight_decay"),
        "OPTIM.MOMENTUM": run_args.get("momentum"),
        "OPTIM.NESTEROV": run_args.get("nesterov"),
        "TRAIN.BATCH_SIZE": run_args.get("train_batch_size"),
        "TEST.BATCH_SIZE": run_args.get("test_batch_size"),
        "ACTIVE_LEARNING.MAX_DELTA": run_args.get("max_delta"),
        "ACTIVE_LEARNING.DELTA_RESOLUTION": run_args.get("delta_resolution"),
        "ACTIVE_LEARNING.DROPOUT_ITERATIONS": run_args.get("dropout_iterations"),
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
        "ACTIVE_LEARNING.NAS_FILTER_METHOD": run_args.get("nas_filter_method"),
        "ACTIVE_LEARNING.NAS_CV_FOLDS": run_args.get("nas_cv_folds"),
        "ACTIVE_LEARNING.NAS_CV_MAX_ITER": run_args.get("nas_cv_max_iter"),
        "ACTIVE_LEARNING.NAS_KNN_K": run_args.get("nas_knn_k"),
        "ACTIVE_LEARNING.NAS_AUM_EPOCHS": run_args.get("nas_aum_epochs"),
        "ACTIVE_LEARNING.NAS_AUM_FAKE_RATIO": run_args.get("nas_aum_fake_ratio"),
        "ACTIVE_LEARNING.NAS_AUM_FAKE_PERCENTILE": run_args.get("nas_aum_fake_percentile"),
        "ACTIVE_LEARNING.NAS_AUM_LR": run_args.get("nas_aum_lr"),
        "ACTIVE_LEARNING.NAS_AUM_WEIGHT_DECAY": run_args.get("nas_aum_weight_decay"),
        "ACTIVE_LEARNING.NAS_AUM_MAX_PRED_NOISY_RATE": run_args.get("nas_aum_max_pred_noisy_rate"),
        "ACTIVE_LEARNING.NAS_MIN_LABELED_SIZE": run_args.get("nas_min_labeled_size"),
        "ACTIVE_LEARNING.NAS_NOISE_DROPOUT": run_args.get("nas_noise_dropout"),
        "ACTIVE_LEARNING.NAS_NOISE_DROPOUT_MIN_FRACTION": run_args.get("nas_noise_dropout_min_fraction"),
    }
    _apply_eval_model_alias(args_dict, run_args.get("eval_model_type"))

    cfg = initialize_cfg(cfg_file, {k: v for k, v in args_dict.items() if v is not None})
    _set_new_allowed(cfg)
    for k, v in args_dict.items():
        _set_cfg(cfg, k, v)
    apply_paper_hparams(cfg, pipeline)
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
    if noise_type == "cifar10n":
        patterns = ["**/*CIFAR-10*.pt", "**/*CIFAR10*.pt"]
    elif noise_type == "cifar100n":
        patterns = ["**/*CIFAR-100*.pt", "**/*CIFAR100*.pt"]
    else:
        raise ValueError(f"Unsupported CIFAR-N noise_type={noise_type}")

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
    """Return the currently active labels used for training/evaluation."""
    if hasattr(dataset, "noisy_labels"):
        return np.asarray(dataset.noisy_labels, dtype=np.int64)
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "targets"):
        return np.asarray(dataset.dataset.targets, dtype=np.int64)
    return None


def _get_clean_targets(dataset: Any) -> Optional[np.ndarray]:
    """Return the original clean labels; never prefer .noisy_labels here."""
    if hasattr(dataset, "clean_labels"):
        return np.asarray(dataset.clean_labels, dtype=np.int64)
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "targets"):
        return np.asarray(dataset.dataset.targets, dtype=np.int64)
    return None


def _set_targets(dataset: Any, labels: np.ndarray) -> None:
    """Force a dataset/wrapper to expose the active labels in .targets.

    Several AL methods, especially DCoM, directly access train_data.targets.
    Synthetic NoisyDataset wrappers may keep noisy labels elsewhere, so we also
    attach .targets to the wrapper itself.
    """
    labels = np.asarray(labels, dtype=np.int64)
    dataset.targets = labels.tolist() if isinstance(getattr(dataset, "targets", None), list) else labels.copy()
    if hasattr(dataset, "noisy_labels"):
        dataset.noisy_labels = labels.copy()


def _ensure_noisy_dataset_label_attrs(dataset: Any, clean_labels: np.ndarray) -> Any:
    """Make noisy/clean label fields consistent for wrappers and DCoM."""
    clean_labels = np.asarray(clean_labels, dtype=np.int64)

    # Always force the clean reference. Do not trust an existing clean_labels
    # field from a wrapper or previous object.
    dataset.clean_labels = clean_labels.copy()

    labels = _get_targets(dataset)
    if labels is None:
        raise RuntimeError("Noisy dataset has no readable labels after noise injection")
    labels = np.asarray(labels, dtype=np.int64)

    if not hasattr(dataset, "noisy_labels"):
        dataset.noisy_labels = labels.copy()
    _set_targets(dataset, labels)
    dataset.is_noisy = labels != clean_labels
    return dataset


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
    ds_key = _dataset_key(cfg)
    if noise_type == "cifar10n" and (num_classes != 10 or ds_key != "CIFAR10"):
        raise ValueError("cifar10n requires MODEL.NUM_CLASSES=10 and a CIFAR10 config")
    if noise_type == "cifar100n" and (num_classes != 100 or ds_key != "CIFAR100"):
        raise ValueError("cifar100n requires MODEL.NUM_CLASSES=100 and a CIFAR100 config")
    if noise_type not in {"cifar10n", "cifar100n"}:
        raise ValueError(f"Unsupported CIFAR-N noise_type={noise_type}")


def _resolve_transition_matrix(cfg: Any, pipeline: Dict[str, Any], noise_type: str, noise_rate: float) -> tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """Create/load the transition matrix used by synthetic noise."""
    num_classes = int(cfg.MODEL.NUM_CLASSES)
    dataset_name = _cfg_get(cfg, 'DATASET.NAME') or _dataset_key(cfg)
    transition_path = pipeline.get('noise', {}).get('transition_matrix')
    superclass_groups_path = pipeline.get('noise', {}).get('superclass_groups_path')

    meta: Dict[str, Any] = {
        'noise_type': noise_type,
        'noise_rate': float(noise_rate),
        'num_classes': num_classes,
        'dataset_name': dataset_name,
        'provided_transition_path': transition_path,
        'provided_superclass_groups_path': superclass_groups_path,
    }

    if noise_type not in {'sym', 'symmetric', 'asym', 'asymmetric'} and transition_path is None:
        return None, meta

    if transition_path is not None:
        meta['source'] = 'provided_transition_path'
    elif noise_type in {'sym', 'symmetric'}:
        meta['source'] = 'generated_symmetric'
    elif num_classes == 10:
        meta['source'] = 'generated_cifar10_asymmetric_common_flips'
    elif superclass_groups_path is not None:
        meta['source'] = 'provided_cifar100_superclass_groups'
    else:
        meta['source'] = 'torchvision_cifar100_raw_coarse_labels'

    transition = make_transition_matrix(
        noise_type=noise_type,
        noise_rate=noise_rate,
        num_classes=num_classes,
        dataset_name=dataset_name,
        transition_path=transition_path,
        superclass_groups_path=superclass_groups_path,
        data_root=cfg.DATASET.ROOT_DIR,
    )

    if transition is not None:
        resolved_path = os.path.join(cfg.EXP_DIR, f"resolved_transition_{noise_type}_{noise_rate:.3f}.npy")
        save_transition_matrix(transition, resolved_path)
        meta['resolved_transition_path'] = resolved_path
        meta['row_sum_min'] = float(np.min(transition.sum(axis=1)))
        meta['row_sum_max'] = float(np.max(transition.sum(axis=1)))
        meta['diagonal_mean'] = float(np.mean(np.diag(transition)))
        meta['off_diagonal_mass_mean'] = float(np.mean(1.0 - np.diag(transition)))
        pipeline.setdefault('noise', {})['resolved_transition_matrix_path'] = resolved_path
        pipeline.setdefault('noise', {})['transition_metadata'] = meta

    return transition, meta

def apply_noise(clean_train_data: Any, cfg: Any, pipeline: Dict[str, Any]) -> tuple[Any, bool]:
    noise_type = pipeline["noise"].get("type")
    noise_rate = float(pipeline["noise"].get("rate") or 0.0)
    if noise_type in [None, "", "clean"] or noise_rate == 0.0 and not str(noise_type).lower().startswith("cifar"):
        return clean_train_data, False

    noise_type = str(noise_type).lower()
    clean_labels = _get_clean_targets(clean_train_data)
    if clean_labels is None:
        raise RuntimeError("Could not read clean labels from train dataset")

    if noise_type.startswith("cifar"):
        _validate_cifar_n_config(cfg, noise_type)
        variant = pipeline["noise"].get("variant", "auto")
        explicit_pt_path = pipeline["noise"].get("labels_path")

        if explicit_pt_path:
            pt_path = explicit_pt_path
            if not os.path.exists(pt_path):
                raise FileNotFoundError(f"CIFAR-N labels path does not exist: {pt_path}")
        else:
            extracted = _download_and_extract_cifar_n(cfg.DATASET.ROOT_DIR)
            if extracted is None:
                raise RuntimeError("Could not download/extract CIFAR-N labels")
            pt_path = _find_cifar_n_pt(extracted, noise_type)
            if pt_path is None:
                raise FileNotFoundError(f"Could not find .pt file for {noise_type} under {extracted}")

        noisy_labels = load_cifar_n_labels(pt_path, variant)
        if len(noisy_labels) != len(clean_labels):
            raise ValueError(f"CIFAR-N length mismatch: {len(noisy_labels)} labels vs {len(clean_labels)} samples")

        noisy_train_data = copy.copy(clean_train_data)
        noisy_train_data.clean_labels = clean_labels.copy()
        noisy_train_data.noisy_labels = noisy_labels.copy()
        noisy_train_data = _ensure_noisy_dataset_label_attrs(noisy_train_data, clean_labels)
        try:
            summary = noise_summary(clean_labels, get_current_train_labels(noisy_train_data))
            summary["cifar_n_pt_path"] = str(pt_path)
            summary["cifar_n_variant"] = str(variant)
            append_to_single_run_log(cfg, "NOISE SUMMARY", summary)
        except Exception as exc:
            logger.warning("Could not build CIFAR-N noise summary: %s", exc)
        logger.info("Applied %s labels from %s | noisy rate in file: %.4f", noise_type, pt_path, noisy_train_data.is_noisy.mean())
        return noisy_train_data, True

    transition, transition_meta = _resolve_transition_matrix(cfg, pipeline, noise_type, noise_rate)
    if transition_meta is not None:
        append_to_single_run_log(cfg, "NOISE TRANSITION MATRIX", transition_meta)

    noisy_train_data = NoisyDataset(
        copy.copy(clean_train_data),
        noise_type=noise_type,
        noise_rate=noise_rate,
        num_classes=int(cfg.MODEL.NUM_CLASSES),
        seed=cfg.RNG_SEED,
        noise_transition_matrix=transition,
    )
    noisy_train_data = _ensure_noisy_dataset_label_attrs(noisy_train_data, clean_labels)
    try:
        summary = noise_summary(clean_labels, get_current_train_labels(noisy_train_data))
        append_to_single_run_log(cfg, "NOISE SUMMARY", summary)
    except Exception as exc:
        logger.warning("Could not build noise summary: %s", exc)
    logger.info("Applied synthetic noise: type=%s rate=%.3f actual=%.4f", noise_type, noise_rate, float(noisy_train_data.is_noisy.mean()))
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




def _cfg_get(cfg: Any, dotted_key: str, default: Any = None) -> Any:
    node = cfg
    for part in dotted_key.split('.'):
        try:
            node = getattr(node, part)
        except Exception:
            return default
    return node


def _maybe_bool(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _method_runtime_details(cfg: Any, pipeline: Dict[str, Any], lSet: Optional[np.ndarray] = None,
                            uSet: Optional[np.ndarray] = None, activeSet: Optional[np.ndarray] = None) -> Dict[str, Any]:
    al_name = _normalize_al_name(_cfg_get(cfg, 'ACTIVE_LEARNING.SAMPLING_FN', pipeline.get('active_learning', {}).get('method')))
    budget = int(_cfg_get(cfg, 'ACTIVE_LEARNING.BUDGET_SIZE', pipeline.get('active_learning', {}).get('budget') or 0))
    details: Dict[str, Any] = {
        'method': al_name,
        'budget': budget,
        'max_iter': _cfg_get(cfg, 'ACTIVE_LEARNING.MAX_ITER'),
        'fine_tune': _maybe_bool(_cfg_get(cfg, 'ACTIVE_LEARNING.FINE_TUNE')),
    }

    if lSet is not None:
        details['lSet_size_before_sampling'] = int(len(lSet))
    if uSet is not None:
        details['uSet_size_before_sampling'] = int(len(uSet))
    if activeSet is not None:
        details['activeSet_size'] = int(len(activeSet))

    if al_name.startswith('typiclust'):
        details.update({
            'family': 'TypiClust',
            'features': 'ds_utils.load_features(..., normalized=True) unless typiclust_dc/SCAN path',
            'distance': 'KMeans + FAISS IndexFlatL2 KNN over features',
            'k_nn': 20,
            'max_num_clusters': 500,
            'min_cluster_size': 5,
            'is_scan_dc': al_name.endswith('dc'),
        })
        if lSet is not None:
            details['expected_num_clusters_this_round'] = int(min(len(lSet) + budget, 500))

    elif al_name in {'probcover', 'prob_cover', 'probcover_orig', 'prob_cover_orig', 'maxherding', 'max_herding'}:
        details.update({
            'family': 'CoverageMatrixMethod',
            'features': 'ds_utils.load_features(..., normalized=True)',
            'kernel_type': _cfg_get(cfg, 'KERNEL_TYPE'),
            'sparse_K': _maybe_bool(_cfg_get(cfg, 'SPARSE_K')),
            'initial_delta': _cfg_get(cfg, 'ACTIVE_LEARNING.INITIAL_DELTA'),
            'initial_sigma': _cfg_get(cfg, 'ACTIVE_LEARNING.INITIAL_SIGMA'),
            'matrix_dtype': _cfg_get(cfg, 'MATRIX_DTYPE'),
            'kernel_build_batch_size': _cfg_get(cfg, 'KERNEL_BUILD_BATCH_SIZE'),
        })

    elif al_name in {'maxherding_orig', 'max_herding_orig'}:
        details.update({
            'family': 'MaxHerding original',
            'features': 'ds_utils.load_features(..., normalized=True)',
            'kernel_type': _cfg_get(cfg, 'KERNEL_TYPE'),
            'delta_or_lengthscale': _cfg_get(cfg, 'ACTIVE_LEARNING.INITIAL_DELTA'),
            'initial_sigma': _cfg_get(cfg, 'ACTIVE_LEARNING.INITIAL_SIGMA'),
        })

    elif al_name == 'dcom':
        details.update({
            'family': 'DCoM',
            'features': 'ds_utils.load_features(..., normalized=True)',
            'initial_delta': _cfg_get(cfg, 'ACTIVE_LEARNING.INITIAL_DELTA'),
            'max_delta': _cfg_get(cfg, 'ACTIVE_LEARNING.MAX_DELTA'),
            'delta_resolution': _cfg_get(cfg, 'ACTIVE_LEARNING.DELTA_RESOLUTION'),
            'k_logistic': _cfg_get(cfg, 'ACTIVE_LEARNING.K_LOGISTIC'),
            'a_logistic': _cfg_get(cfg, 'ACTIVE_LEARNING.A_LOGISTIC'),
        })

    elif al_name in {'misp', 'mispc', 'misp_plus', 'all_misp'}:
        details.update({
            'family': 'MISP local',
            'features': 'ds_utils.load_features(..., normalized=True)',
            'kernel_type': _cfg_get(cfg, 'KERNEL_TYPE'),
            'initial_delta': _cfg_get(cfg, 'ACTIVE_LEARNING.INITIAL_DELTA'),
            'alpha': _cfg_get(cfg, 'ALPHA'),
            'diff_method': _cfg_get(cfg, 'DIFF_METHOD'),
            'confidence_method': _cfg_get(cfg, 'CONFIDENCE_METHOD'),
            'cont_method': _cfg_get(cfg, 'CONT_METHOD'),
            'soft_border_val': _cfg_get(cfg, 'SOFT_BORDER_VAL'),
            'norm_importance': _maybe_bool(_cfg_get(cfg, 'NORM_IMPORTANCE')),
        })

    elif al_name == 'coreset':
        details.update({
            'family': 'CoreSet/k-center',
            'features': 'model penultimate features from current checkpoint',
            'distance': 'squared L2 in Sampling.py',
            'explicit_feature_l2_norm': False,
        })

    elif al_name in {'uncertainty', 'entropy', 'margin'}:
        details.update({
            'family': 'prediction uncertainty',
            'score_space': 'model probabilities/logits',
            'feature_normalization_used': False,
        })

    elif al_name in {'bald', 'dbal'}:
        details.update({
            'family': 'MC-dropout uncertainty',
            'dropout_iterations': _cfg_get(cfg, 'ACTIVE_LEARNING.DROPOUT_ITERATIONS'),
            'score_space': 'model probabilities with dropout',
        })

    elif al_name == 'vaal':
        details.update({
            'family': 'VAAL',
            'z_dim': _cfg_get(cfg, 'VAAL.Z_DIM'),
            'vae_bs': _cfg_get(cfg, 'VAAL.VAE_BS'),
            'vae_epochs': _cfg_get(cfg, 'VAAL.VAE_EPOCHS'),
            'vae_lr': _cfg_get(cfg, 'VAAL.VAE_LR'),
            'disc_lr': _cfg_get(cfg, 'VAAL.DISC_LR'),
            'beta': _cfg_get(cfg, 'VAAL.BETA'),
            'adversary_param': _cfg_get(cfg, 'VAAL.ADVERSARY_PARAM'),
        })

    return details


def build_run_summary(cfg: Any, pipeline: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'command': ' '.join(sys.argv),
        'cwd': os.getcwd(),
        'pid': os.getpid(),
        'cfg_file': pipeline.get('cfg'),
        'experiment': {
            'exp_dir': _cfg_get(cfg, 'EXP_DIR'),
            'episode_dir': _cfg_get(cfg, 'EPISODE_DIR'),
            'out_dir': _cfg_get(cfg, 'OUT_DIR'),
            'exp_name': _cfg_get(cfg, 'EXP_NAME'),
            'seed': _cfg_get(cfg, 'RNG_SEED'),
        },
        'dataset': {
            'name': _cfg_get(cfg, 'DATASET.NAME'),
            'root_dir': _cfg_get(cfg, 'DATASET.ROOT_DIR'),
            'features_tag': _cfg_get(cfg, 'DATASET.FEATURES_TAG'),
            'val_ratio': _cfg_get(cfg, 'DATASET.VAL_RATIO'),
            'eval_model_type': _cfg_get(cfg, 'EVAL_MODEL_TYPE'),
        },
        'model': {
            'type': _cfg_get(cfg, 'MODEL.TYPE'),
            'num_classes': _cfg_get(cfg, 'MODEL.NUM_CLASSES'),
            'loss_fun': _cfg_get(cfg, 'MODEL.LOSS_FUN'),
            'gce_q': _cfg_get(cfg, 'MODEL.GCE_Q'),
            'bootstrap_beta': _cfg_get(cfg, 'MODEL.BOOTSTRAP_BETA'),
            'use_1nn': _maybe_bool(_cfg_get(cfg, 'MODEL.USE_1NN')),
            'linear_from_features': _maybe_bool(_cfg_get(cfg, 'MODEL.LINEAR_FROM_FEATURES')),
            'linear_from_images': _maybe_bool(_cfg_get(cfg, 'MODEL.LINEAR_FROM_IMAGES')),
        },
        'training': {
            'optimizer': _cfg_get(cfg, 'OPTIM.TYPE'),
            'max_epoch': _cfg_get(cfg, 'OPTIM.MAX_EPOCH'),
            'base_lr': _cfg_get(cfg, 'OPTIM.BASE_LR'),
            'min_lr': _cfg_get(cfg, 'OPTIM.MIN_LR'),
            'lr_policy': _cfg_get(cfg, 'OPTIM.LR_POLICY'),
            'weight_decay': _cfg_get(cfg, 'OPTIM.WEIGHT_DECAY'),
            'momentum': _cfg_get(cfg, 'OPTIM.MOMENTUM'),
            'nesterov': _maybe_bool(_cfg_get(cfg, 'OPTIM.NESTEROV')),
            'train_batch_size': _cfg_get(cfg, 'TRAIN.BATCH_SIZE'),
            'test_batch_size': _cfg_get(cfg, 'TEST.BATCH_SIZE'),
            'num_workers': _cfg_get(cfg, 'DATA_LOADER.NUM_WORKERS'),
            'pin_memory': _maybe_bool(_cfg_get(cfg, 'DATA_LOADER.PIN_MEMORY')),
        },
        'active_learning': _method_runtime_details(cfg, pipeline),
        'noise': {
            'type': pipeline.get('noise', {}).get('type'),
            'rate': pipeline.get('noise', {}).get('rate'),
            'transition_matrix': pipeline.get('noise', {}).get('transition_matrix'),
            'resolved_transition_matrix_path': pipeline.get('noise', {}).get('resolved_transition_matrix_path'),
            'transition_metadata': pipeline.get('noise', {}).get('transition_metadata'),
            'superclass_groups_path': pipeline.get('noise', {}).get('superclass_groups_path'),
            'cifar_n_variant': pipeline.get('noise', {}).get('variant'),
        },
        'noise_handling': pipeline.get('noise_handling', {}),
        'extra_evaluation': pipeline.get('evaluation', {}),
        'paper_hparams': {
            'enabled': bool(pipeline.get('paper_hparams', False)),
            'applied': pipeline.get('paper_hparams_applied', {}),
        },
        'raw_pipeline': pipeline,
    }
    if '_runtime_cleaning_state' in summary.get('raw_pipeline', {}):
        summary['raw_pipeline'] = dict(summary['raw_pipeline'])
        summary['raw_pipeline'].pop('_runtime_cleaning_state', None)
    if extra:
        summary['runtime'] = _jsonify(extra)
    return _jsonify(summary)


def _single_run_log_path(cfg: Any) -> str:
    return os.path.join(cfg.EXP_DIR, "run_full_log.txt")


def append_to_single_run_log(cfg: Any, title: str, obj: Any) -> None:
    """Append one structured section to the single experiment log."""
    path = _single_run_log_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = json.dumps(_jsonify(obj), indent=2, sort_keys=True)
    header = "=" * 30 + f" {title} " + "=" * 30
    with open(path, "a") as f:
        f.write("\n" + header + "\n")
        f.write(text + "\n")
        f.write("=" * 80 + "\n")
    logger.info("\n%s\n%s\n%s", header, text, "=" * 80)


def log_run_summary(cfg: Any, pipeline: Dict[str, Any], title: str,
                    extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    summary = build_run_summary(cfg, pipeline, extra=extra)
    append_to_single_run_log(cfg, title, summary)
    print(f"Updated single run log: {_single_run_log_path(cfg)}", flush=True)
    return summary


def log_episode_state(cfg: Any, pipeline: Dict[str, Any], cur_episode: int, lSet: np.ndarray,
                      uSet: np.ndarray, valSet: Optional[np.ndarray], stage: str,
                      activeSet: Optional[np.ndarray] = None) -> Dict[str, Any]:
    state = {
        'episode': int(cur_episode),
        'stage': stage,
        'lSet_size': int(len(lSet)),
        'uSet_size': int(len(uSet)),
        'valSet_size': int(len(valSet)) if valSet is not None else None,
        'method_details': _method_runtime_details(cfg, pipeline, lSet=lSet, uSet=uSet, activeSet=activeSet),
    }
    if activeSet is not None and len(activeSet) > 0:
        state['activeSet_preview'] = [int(x) for x in np.asarray(activeSet[:10]).tolist()]
    state = _jsonify(state)
    logger.info("EPISODE_STATE %s", json.dumps(state, sort_keys=True))
    append_to_single_run_log(cfg, f"EPISODE_STATE_{cur_episode}_{stage}", state)
    return state

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


def active_sampling_part(
    cfg,
    checkpoint_file,
    cur_episode,
    data_obj,
    lSet,
    train_data,
    uSet,
    valSet,
    clean_val_data,
    al_obj,
    train_al_module,
    pipeline,
    npc_filter_results,
):
    logger.info("======== ACTIVE SAMPLING ========")
    log_episode_state(cfg, pipeline, cur_episode, lSet, uSet, valSet, "before_active_sampling")
    clf_model = train_al_module.mh.get_model(cfg, checkpoint_file)
    activeSet, new_uSet = al_obj.sample_from_uSet(
        clf_model,
        lSet,
        uSet,
        train_data,
        checkpoint_file=checkpoint_file,
    )

    algorithm = getattr(al_obj, "al_algorithm", None)
    npc_stats = getattr(algorithm, "last_npc_filter_stats", None)
    if isinstance(npc_stats, dict):
        npc_entry = {"episode": int(cur_episode), **copy.deepcopy(npc_stats)}
        npc_filter_results.append(npc_entry)
        append_to_single_run_log(
            cfg,
            f"NPC_FILTER_STATS_EPISODE_{cur_episode}",
            npc_entry,
        )
        logger.info("NPC_FILTER_STATS %s", json.dumps(_jsonify(npc_entry), sort_keys=True))

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
    log_episode_state(cfg, pipeline, cur_episode, lSet, uSet, valSet, "after_active_sampling", activeSet=activeSet)
    return lSet, lSet_loader, uSet, valSet_loader


def run_label_cleaning(train_data: Any, lSet: np.ndarray, cfg: Any, checkpoint_file: Optional[str], train_al_module: Any,
                       threshold: float = 0.9, pred_source: str = "checkpoint", crossval_folds: int = 5,
                       crossval_max_iter: int = 300) -> Dict[str, Any]:
    if pred_source == "crossval":
        return noise_cleaning.run_crossval_cleaning(
            train_data,
            lSet,
            cfg,
            threshold=threshold,
            folds=crossval_folds,
            max_iter=crossval_max_iter,
        )

    return noise_cleaning.run_checkpoint_cleaning(
        train_data,
        lSet,
        cfg,
        checkpoint_file,
        train_al_module,
        threshold=threshold,
    )


def run_aum(train_data: Any, lSet: np.ndarray, cfg: Any, checkpoint_file: Optional[str], train_al_module: Any,
            pipeline: Dict[str, Any]) -> Dict[str, Any]:
    state = pipeline.setdefault("_runtime_cleaning_state", {})
    return noise_cleaning.run_aum_cleaning(
        train_data,
        lSet,
        cfg,
        checkpoint_file,
        train_al_module,
        state,
        percentile=float(pipeline["noise_handling"].get("aum_percentile", 10.0)),
        confidence_threshold=float(pipeline["noise_handling"].get("aum_conf_threshold", 0.8)),
    )


def run_temporal(train_data: Any, lSet: np.ndarray, cfg: Any, checkpoint_file: Optional[str], train_al_module: Any,
                 pipeline: Dict[str, Any]) -> Dict[str, Any]:
    state = pipeline.setdefault("_runtime_cleaning_state", {})
    return noise_cleaning.run_temporal_cleaning(
        train_data,
        lSet,
        cfg,
        checkpoint_file,
        train_al_module,
        state,
        threshold=float(pipeline["noise_handling"].get("temporal_conf_threshold", 0.8)),
        momentum=float(pipeline["noise_handling"].get("temporal_momentum", 0.8)),
        min_seen=int(pipeline["noise_handling"].get("temporal_min_seen", 2)),
    )


def run_noise_method(method: Optional[str], train_data: Any, lSet: np.ndarray, cfg: Any, checkpoint_file: Optional[str],
                     train_al_module: Any, cur_episode: int, final_val_acc: float, pipeline: Dict[str, Any]) -> Dict[str, Any]:
    method = "none" if method in [None, ""] else str(method).lower()
    if method not in SUPPORTED_NOISE_METHODS:
        raise NotImplementedError(f"noise_method={method} is not implemented. Use: {sorted(x for x in SUPPORTED_NOISE_METHODS if x)}")
    if method == "none":
        return {"status": "skipped", "method": "none"}

    warmup_episode = int(pipeline["noise_handling"].get("warmup_episode", 3))
    warmup_min_acc = float(pipeline["noise_handling"].get("warmup_min_acc", 60.0))
    if cur_episode < warmup_episode and final_val_acc < warmup_min_acc:
        return {"status": "skipped", "reason": "warmup", "method": method}

    threshold = float(pipeline["noise_handling"].get("cleaning_conf_threshold", 0.9))
    pred_source = str(pipeline["noise_handling"].get("cleaning_pred_source", "checkpoint")).lower()
    crossval_folds = int(pipeline["noise_handling"].get("crossval_folds", 5))
    crossval_max_iter = int(pipeline["noise_handling"].get("crossval_max_iter", 300))

    if method == "label_cleaning":
        return run_label_cleaning(
            train_data, lSet, cfg, checkpoint_file, train_al_module,
            threshold=threshold,
            pred_source=pred_source,
            crossval_folds=crossval_folds,
            crossval_max_iter=crossval_max_iter,
        )

    if method in {"crossval", "crossval_label_cleaning"}:
        return run_label_cleaning(
            train_data, lSet, cfg, checkpoint_file, train_al_module,
            threshold=threshold,
            pred_source="crossval",
            crossval_folds=crossval_folds,
            crossval_max_iter=crossval_max_iter,
        )

    if method in {"aum", "aum_relabel"}:
        return run_aum(train_data, lSet, cfg, checkpoint_file, train_al_module, pipeline)

    if method == "temporal":
        return run_temporal(train_data, lSet, cfg, checkpoint_file, train_al_module, pipeline)

    raise NotImplementedError(method)


def _collect_dataset_numpy(dataset: Any, indices: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """Collect dataset items directly, without DataLoader drop_last."""
    xs, ys = [], []

    old_no_aug = getattr(dataset, "no_aug", None)
    if hasattr(dataset, "no_aug"):
        dataset.no_aug = True

    if indices is None:
        indices = np.arange(len(dataset))

    try:
        for idx in indices:
            xb, yb = dataset[int(idx)]

            if hasattr(xb, "detach"):
                xb = xb.detach().cpu().numpy()
            else:
                xb = np.asarray(xb)

            if hasattr(yb, "detach"):
                yb = yb.detach().cpu().numpy()
            else:
                yb = np.asarray(yb)

            xs.append(xb)
            ys.append(yb)
    finally:
        if old_no_aug is not None:
            dataset.no_aug = old_no_aug

    x = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)

    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    return x, y


def evaluate_final_knn(
    train_data: Any,
    test_data: Any,
    lSet: np.ndarray,
    valSet: np.ndarray,
    *,
    k: int = 1,
) -> Dict[str, Any]:
    """Fit KNN on the full final lSet and evaluate full val/test.

    Do not use DataLoaders here, because training loaders may use drop_last=True.
    In the previous version this caused train_size=128 even when final_lSet_len=200.
    """
    from sklearn.neighbors import KNeighborsClassifier

    train_x, train_y = _collect_dataset_numpy(train_data, lSet)
    val_x, val_y = _collect_dataset_numpy(train_data, valSet)
    test_x, test_y = _collect_dataset_numpy(test_data, None)
    train_x = noise_cleaning.l2_normalize(train_x)
    val_x = noise_cleaning.l2_normalize(val_x)
    test_x = noise_cleaning.l2_normalize(test_x)

    if len(train_y) == 0:
        return {"status": "skipped", "reason": "empty labeled set"}

    k_eff = int(min(max(1, k), len(train_y)))
    clf = KNeighborsClassifier(n_neighbors=k_eff, metric="euclidean", n_jobs=-1)

    start = time.time()
    clf.fit(train_x, train_y)
    fit_time = time.time() - start

    def _acc(x: np.ndarray, y: np.ndarray) -> float:
        preds = clf.predict(x)
        return float(100.0 * (preds == y).mean())

    start = time.time()
    result = {
        "status": "completed",
        "k_requested": int(k),
        "k_effective": int(k_eff),
        "metric": "euclidean",
        "train_size": int(train_x.shape[0]),
        "val_size": int(val_x.shape[0]),
        "test_size": int(test_x.shape[0]),
        "feature_dim": int(train_x.shape[1]),
        "train_acc": _acc(train_x, train_y),
        "val_acc": _acc(val_x, val_y),
        "test_acc": _acc(test_x, test_y),
        "fit_time_sec": float(fit_time),
    }
    result["eval_time_sec"] = float(time.time() - start)
    return result




def _nas_filter_from_al_method(al_method: Optional[str]) -> Optional[str]:
    name = "" if al_method is None else str(al_method).lower().replace("-", "_")
    if name == "npc_probcover_crossval":
        return "crossval"
    if name == "npc_probcover_knn":
        return "knn"
    if name == "npc_probcover_aum":
        return "aum"
    if name == "npc_probcover_oracle":
        return "oracle"
    return None


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
            "superclass_groups_path": args.superclass_groups_path,
            "variant": args.cifar_n_variant,
            "labels_path": args.cifar_n_labels_path,
        },
        "active_learning": {
            "method": al_method,
            "budget": args.budget,
            "max_iter": args.max_iter,
            "nas_filter_method": args.nas_filter_method or _nas_filter_from_al_method(al_method),
            "nas_noise_dropout": bool(args.nas_noise_dropout),
            "nas_noise_dropout_min_fraction": float(args.nas_noise_dropout_min_fraction),
            "nas_min_labeled_size": int(args.nas_min_labeled_size),
            "nas_aum_max_pred_noisy_rate": float(args.nas_aum_max_pred_noisy_rate),
        },
        "noise_handling": {
            "method": noise_method_norm,
            "warmup_episode": args.warmup_episode,
            "warmup_min_acc": args.warmup_min_acc,
            "cleaning_conf_threshold": args.cleaning_conf_threshold,
            "cleaning_pred_source": args.cleaning_pred_source,
            "crossval_folds": args.crossval_folds,
            "crossval_max_iter": args.crossval_max_iter,
            "temporal_momentum": args.temporal_momentum,
            "temporal_min_seen": args.temporal_min_seen,
            "temporal_conf_threshold": args.temporal_conf_threshold,
            "aum_percentile": args.aum_percentile,
            "aum_conf_threshold": args.aum_conf_threshold,
        },
        "evaluation": {
            "also_eval_knn": bool(args.also_eval_knn),
            "knn_k": int(args.knn_k),
        },
        "paper_hparams": args.paper_hparams,
        "run_args": {
            "seed": args.seed,
            "initial_size": args.initial_size,
            "exp_name": exp_name,
            "features_tag": args.features_tag,
            "initial_delta": args.initial_delta,
            "initial_sigma": args.initial_sigma,
            "kernel_type": args.kernel_type,
            "eval_model_type": args.eval_model_type,
            "loss_fun": args.loss_fun,
            "gce_q": args.gce_q,
            "bootstrap_beta": args.bootstrap_beta,
            "max_epoch": args.max_epoch,
            "base_lr": args.base_lr,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "nesterov": args.nesterov,
            "train_batch_size": args.train_batch_size,
            "test_batch_size": args.test_batch_size,
            "max_delta": args.max_delta,
            "delta_resolution": args.delta_resolution,
            "dropout_iterations": args.dropout_iterations,
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
            "nas_filter_method": args.nas_filter_method,
            "nas_cv_folds": args.nas_cv_folds,
            "nas_cv_max_iter": args.nas_cv_max_iter,
            "nas_knn_k": args.nas_knn_k,
            "nas_aum_epochs": args.nas_aum_epochs,
            "nas_aum_fake_ratio": args.nas_aum_fake_ratio,
            "nas_aum_fake_percentile": args.nas_aum_fake_percentile,
            "nas_aum_lr": args.nas_aum_lr,
            "nas_aum_weight_decay": args.nas_aum_weight_decay,
            "nas_aum_max_pred_noisy_rate": args.nas_aum_max_pred_noisy_rate,
            "nas_min_labeled_size": args.nas_min_labeled_size,
            "nas_noise_dropout": args.nas_noise_dropout,
            "nas_noise_dropout_min_fraction": args.nas_noise_dropout_min_fraction,
        },
    }


def run_one(pipeline: Dict[str, Any]) -> Dict[str, Any]:
    from deep_al.tools import train_al as train_al_module

    cfg = build_cfg(pipeline["cfg"], pipeline)

    # train_al.py calls losses.get_loss_fun() without passing cfg.
    # Force its losses module to use this run-specific cloned cfg.
    try:
        if hasattr(train_al_module.losses, "set_runtime_cfg"):
            train_al_module.losses.set_runtime_cfg(cfg)
        else:
            train_al_module.losses.cfg = cfg
    except Exception:
        logger.exception("Failed to set runtime cfg for losses module")

    if cfg.RNG_SEED is None:
        cfg.RNG_SEED = int(np.random.randint(100000))
    setup_experiment_directory(cfg)
    dump_cfg(cfg)
    train_al_module.lu.setup_logging(cfg)

    logger.info("Experiment dir: %s", cfg.EXP_DIR)
    append_to_single_run_log(cfg, "PAPER HPARAMS APPLIED", pipeline.get("paper_hparams_applied", {}))

    # First summary: exact config/HPs before data loading.
    log_run_summary(cfg, pipeline, "RUN CONFIG SUMMARY - PRE DATA")

    data_obj = Data(cfg)
    logger.info("Loading datasets from %s", cfg.DATASET.ROOT_DIR)
    print("Loading datasets...", flush=True)
    clean_train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True)
    test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True)
    logger.info("Datasets loaded | train_size=%d test_size=%d", train_size, test_size)

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

    clean_labels = _get_clean_targets(clean_train_data)
    noisy_rate_actual = None
    initial_lset_noisy_rate_actual = None
    initial_lset_noisy_count_actual = None

    if clean_labels is not None and train_labels is not None:
        clean_labels_np = np.asarray(clean_labels, dtype=np.int64)
        train_labels_np = np.asarray(train_labels, dtype=np.int64)

        noisy_mask_actual = clean_labels_np != train_labels_np
        noisy_rate_actual = float(noisy_mask_actual.mean())

        lset_np = np.asarray(lSet, dtype=int)
        initial_lset_noisy_count_actual = int(noisy_mask_actual[lset_np].sum())
        initial_lset_noisy_rate_actual = float(noisy_mask_actual[lset_np].mean())

        print(
            f"NOISE_DEBUG noise_type={pipeline['noise']['type']} "
            f"full_noisy_rate={noisy_rate_actual:.4f} "
            f"initial_lset_noisy_count={initial_lset_noisy_count_actual}/{len(lset_np)} "
            f"initial_lset_noisy_rate={initial_lset_noisy_rate_actual:.4f}",
            flush=True,
        )

    run_extra = {
        "train_size": int(train_size),
        "test_size": int(test_size),
        "initial_lSet_size": int(len(lSet)),
        "initial_uSet_size": int(len(uSet)),
        "valSet_size": int(len(valSet)),
        "lSet_path": lSet_path,
        "uSet_path": uSet_path,
        "valSet_path": valSet_path,
        "initial_label_noise_rate_actual": noisy_rate_actual,
        "initial_lset_noisy_rate_actual": initial_lset_noisy_rate_actual,
        "initial_lset_noisy_count_actual": initial_lset_noisy_count_actual,
        "noisy_wrapper_applied": bool(noisy_applied),
        "method_details_with_initial_pool": _method_runtime_details(cfg, pipeline, lSet=lSet, uSet=uSet),
    }
    log_run_summary(cfg, pipeline, "RUN CONFIG SUMMARY", extra=run_extra)

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
    npc_filter_results = []
    final_train_acc = final_val_acc = final_test_acc = 0.0
    final_knn_eval = None
    checkpoint_file = None

    episode_states = []

    for cur_episode in range(0, cfg.ACTIVE_LEARNING.MAX_ITER + 1):
        logger.info("======== EPISODE %d BEGINS ========", cur_episode)
        cfg.EPISODE_DIR = os.path.join(cfg.EXP_DIR, f"episode_{cur_episode}")
        os.makedirs(cfg.EPISODE_DIR, exist_ok=True)
        episode_states.append(log_episode_state(cfg, pipeline, cur_episode, lSet, uSet, valSet, "episode_start"))

        if cur_episode == 0 and len(lSet) == 0:
            logger.info("Cold start with 0 labels: skipping episode-0 training")
            episode_training_times.append(0.0)
        else:
            try:
                checkpoint_file, train_time, epoch_times, best_train_acc, best_val_acc, test_acc = train_al_module.train_and_eval_model(
                    cfg, cur_episode, lSet_loader, model, optimizer, test_loader, valSet_loader, plotting_manager
                )
            except Exception as exc:
                tb = traceback.format_exc()
                logger.exception("Training failed")
                append_to_single_run_log(cfg, "TRAINING FAILED", {"reason": repr(exc), "traceback": tb})
                return {"status": "failed", "reason": repr(exc), "traceback": tb, "exp_dir": cfg.EXP_DIR}
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
                cfg,
                checkpoint_file,
                cur_episode,
                data_obj,
                lSet,
                train_data,
                uSet,
                valSet,
                clean_train_data,
                al_obj,
                train_al_module,
                pipeline,
                npc_filter_results,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception("Active sampling failed")
            append_to_single_run_log(cfg, "ACTIVE SAMPLING FAILED", {"reason": repr(exc), "traceback": tb})
            return {"status": "failed", "reason": repr(exc), "traceback": tb, "exp_dir": cfg.EXP_DIR}
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
        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception("Noise handling failed")
            append_to_single_run_log(cfg, "NOISE HANDLING FAILED", {"reason": repr(exc), "traceback": tb})
            return {"status": "failed", "reason": repr(exc), "traceback": tb, "exp_dir": cfg.EXP_DIR}

        if not cfg.ACTIVE_LEARNING.FINE_TUNE:
            model = train_al_module.mh.get_model(cfg, model_init_state=model_init_state)
            optimizer = train_al_module.optim.construct_optimizer(cfg, model, opt_init_state=opt_init_state)

        if checkpoint_file is not None:
            try:
                os.remove(checkpoint_file)
            except OSError:
                pass

    if pipeline.get("evaluation", {}).get("also_eval_knn", False):
        try:
            knn_k = int(pipeline.get("evaluation", {}).get("knn_k", 1))
            logger.info("Running final additional KNN evaluation with k=%d", knn_k)
            final_knn_eval = evaluate_final_knn(train_data, test_data, lSet, valSet, k=knn_k)
            append_to_single_run_log(cfg, "FINAL KNN EVALUATION", final_knn_eval)
            logger.info("Final KNN evaluation: %s", json.dumps(_jsonify(final_knn_eval), sort_keys=True))
        except Exception:
            logger.exception("Final additional KNN evaluation failed")
            final_knn_eval = {"status": "failed", "reason": "exception during final KNN evaluation"}
            append_to_single_run_log(cfg, "FINAL KNN EVALUATION", final_knn_eval)

    final_extra = {
        "train_size": int(train_size),
        "test_size": int(test_size),
        "final_lSet_len": int(len(lSet)),
        "final_uSet_len": int(len(uSet)),
        "final_valSet_len": int(len(valSet)),
        "noisy_wrapper_applied": bool(noisy_applied),
        "initial_label_noise_rate_actual": noisy_rate_actual,
        "initial_lset_noisy_rate_actual": initial_lset_noisy_rate_actual,
        "initial_lset_noisy_count_actual": initial_lset_noisy_count_actual,
        "final_train_acc": float(final_train_acc),
        "final_val_acc": float(final_val_acc),
        "final_test_acc": float(final_test_acc),
        "final_knn_eval": final_knn_eval,
        "effective_loss_fun": str(cfg.MODEL.LOSS_FUN),
        "effective_gce_q": float(getattr(cfg.MODEL, "GCE_Q", 0.7)),
        "effective_bootstrap_beta": float(getattr(cfg.MODEL, "BOOTSTRAP_BETA", 0.8)),
        "episode_training_times": episode_training_times,
        "episode_al_times": episode_al_times,
        "noise_method_results": noise_method_results,
        "npc_filter_results": npc_filter_results,
        "episode_states": episode_states,
    }
    log_run_summary(cfg, pipeline, "FINAL RUN SUMMARY", extra=final_extra)

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
            "initial_label_noise_rate_actual": noisy_rate_actual,
            "initial_lset_noisy_rate_actual": initial_lset_noisy_rate_actual,
            "initial_lset_noisy_count_actual": initial_lset_noisy_count_actual,
            "final_train_acc": float(final_train_acc),
            "final_val_acc": float(final_val_acc),
            "final_test_acc": float(final_test_acc),
            "final_knn_eval": final_knn_eval,
            "effective_loss_fun": str(cfg.MODEL.LOSS_FUN),
            "effective_gce_q": float(getattr(cfg.MODEL, "GCE_Q", 0.7)),
            "effective_bootstrap_beta": float(getattr(cfg.MODEL, "BOOTSTRAP_BETA", 0.8)),
            "episode_training_times": episode_training_times,
            "episode_al_times": episode_al_times,
            "noise_method_results": noise_method_results,
            "npc_filter_results": npc_filter_results,
            "episode_states": episode_states,
            "paper_hparams_applied": pipeline.get("paper_hparams_applied", {}),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run noisy-label Active Learning experiments")
    parser.add_argument("--cfg", dest="cfg_file", required=True, type=str)

    parser.add_argument("--noise_type", default=None, type=str)
    parser.add_argument("--noise_rate", default=0.0, type=float)
    parser.add_argument("--noise_transition", "--noise_transition_path", dest="noise_transition", default=None, type=str,
                        help="Optional path to transition matrix .npy/.json/.csv")
    parser.add_argument("--superclass_groups_path", default=None, type=str,
                        help="Optional CIFAR100 superclass groups JSON/NPY for asymmetric noise. If omitted, torchvision CIFAR100 coarse labels are used.")
    parser.add_argument("--cifar_n_variant", choices=sorted(CIFAR_N_VARIANTS), default="auto", type=str)
    parser.add_argument("--cifar_n_labels_path", default=None, type=str,
                        help="Optional explicit path to CIFAR-N .pt labels, e.g. data/CIFAR-100_human.pt")

    parser.add_argument("--al_method", default=None, type=str)
    parser.add_argument("--budget", default=None, type=int)
    parser.add_argument("--initial_size", default=None, type=int)
    parser.add_argument("--max_iter", default=None, type=int)
    parser.add_argument("--exp_name", default="auto", type=str)

    parser.add_argument("--noise_method", default="none", type=str)
    parser.add_argument("--warmup_episode", default=3, type=int)
    parser.add_argument("--warmup_min_acc", default=60.0, type=float)
    parser.add_argument("--cleaning_conf_threshold", default=0.9, type=float)
    parser.add_argument("--cleaning_pred_source", default="checkpoint", choices=["checkpoint", "crossval"], type=str,
                        help="Prediction source for label_cleaning: checkpoint model or K-fold cross-val head over frozen features.")
    parser.add_argument("--crossval_folds", default=5, type=int)
    parser.add_argument("--crossval_max_iter", default=300, type=int)
    parser.add_argument("--temporal_momentum", default=0.8, type=float)
    parser.add_argument("--temporal_min_seen", default=2, type=int)
    parser.add_argument("--temporal_conf_threshold", default=0.8, type=float)
    parser.add_argument("--aum_percentile", default=10.0, type=float)
    parser.add_argument("--aum_conf_threshold", default=0.8, type=float)

    parser.add_argument("--grid_noise_types", default=None, type=str)
    parser.add_argument("--grid_al_methods", default=None, type=str)
    parser.add_argument("--grid_noise_methods", default=None, type=str)
    parser.add_argument("--grid_out_dir", default=None, type=str)
    parser.add_argument("--grid_json_path", default=None, type=str,
                        help="Path to one shared aggregate grid JSON file.")
    parser.add_argument("--debug_json", action="store_true",
                        help="Write full old-style JSON. Without this flag, write only light final-result entries.")

    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--features_tag", default=None, type=str)
    parser.add_argument("--initial_delta", default=None, type=float)
    parser.add_argument("--initial_sigma", default=None, type=float)
    parser.add_argument("--kernel_type", default=None, type=str)
    parser.add_argument("--eval_model_type", default=None, type=str)
    parser.add_argument("--loss_fun", default=None, choices=["cross_entropy", "gce", "bootstrap_soft", "bootstrap_hard"], type=str)
    parser.add_argument("--gce_q", default=0.7, type=float)
    parser.add_argument("--bootstrap_beta", default=0.8, type=float)
    parser.add_argument("--max_epoch", default=None, type=int)
    parser.add_argument("--base_lr", default=None, type=float)
    parser.add_argument("--weight_decay", default=None, type=float)
    parser.add_argument("--momentum", default=None, type=float)
    parser.add_argument("--nesterov", dest="nesterov", action="store_true", default=None)
    parser.add_argument("--no_nesterov", dest="nesterov", action="store_false")
    parser.add_argument("--train_batch_size", default=None, type=int)
    parser.add_argument("--test_batch_size", default=None, type=int)
    parser.add_argument("--max_delta", default=None, type=float)
    parser.add_argument("--delta_resolution", default=None, type=float)
    parser.add_argument("--dropout_iterations", default=None, type=int)

    parser.add_argument("--paper_hparams", dest="paper_hparams", action="store_true", default=True)
    parser.add_argument("--no_paper_hparams", dest="paper_hparams", action="store_false")

    parser.add_argument("--sparse_K", dest="sparse_K", action="store_true", default=None)
    parser.add_argument("--no_sparse_K", dest="sparse_K", action="store_false")
    parser.add_argument("--matrix_dtype", choices=["float32", "float16"], default=None, type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--high_budget", action="store_true")
    parser.add_argument("--sparse_ds", action="store_true")
    parser.add_argument("--norm_importance", action="store_true")
    parser.add_argument("--decrease_alpha", action="store_true")
    parser.add_argument("--diff_method", default=None, type=str)
    parser.add_argument("--confidence_method", default=None, type=str)
    parser.add_argument("--cont_method", default=None, type=str)
    parser.add_argument("--alpha", default=None, type=float)
    parser.add_argument("--a_logistic", default=None, type=float)
    parser.add_argument("--k_logistic", default=None, type=int)
    parser.add_argument("--soft_border_val", default=None, type=float)
    parser.add_argument("--nas_filter_method", default=None, choices=["none", "crossval", "knn", "aum", "oracle"], type=str)
    parser.add_argument("--nas_cv_folds", default=3, type=int)
    parser.add_argument("--nas_cv_max_iter", default=300, type=int)
    parser.add_argument("--nas_knn_k", default=-1, type=int)
    parser.add_argument("--nas_aum_epochs", default=40, type=int)
    parser.add_argument("--nas_aum_fake_ratio", default=1.0, type=float)
    parser.add_argument("--nas_aum_fake_percentile", default=80.0, type=float)
    parser.add_argument("--nas_aum_lr", default=0.1, type=float)
    parser.add_argument("--nas_aum_weight_decay", default=0.0, type=float)
    parser.add_argument("--nas_aum_max_pred_noisy_rate", default=0.5, type=float,
                        help="Cap the fraction of labeled samples that AUM can mark noisy before dropout. Use 1.0 for uncapped fake-threshold AUM.")
    parser.add_argument("--nas_min_labeled_size", default=0, type=int,
                        help="Apply NPC filtering only once |L| reaches this size. 0 keeps the original behavior.")
    parser.set_defaults(nas_noise_dropout=True)
    parser.add_argument(
        "--nas_noise_dropout",
        dest="nas_noise_dropout",
        action="store_true",
        help="Enable the Shafir et al. NAS noise-dropout rule during query selection.",
    )
    parser.add_argument(
        "--no_nas_noise_dropout",
        dest="nas_noise_dropout",
        action="store_false",
        help="Disable NAS noise dropout.",
    )
    parser.add_argument(
        "--nas_noise_dropout_min_fraction",
        default=0.1,
        type=float,
        help="Minimum eta in eta=max(min(q_hat,1-q_hat), eta_min).",
    )

    parser.add_argument("--also_eval_knn", action="store_true", help="After regular supervised training, also fit KNN on the final labeled set and evaluate train/val/test.")
    parser.add_argument("--knn_k", default=1, type=int, help="K for the additional final KNN evaluation.")

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


def _sum_noise_cleaning_results(noise_method_results: Any) -> Dict[str, Any]:
    """Aggregate cleaning counters from both old and new result formats.

    Supported shapes:
      {"episode": i, "result": {...}}
      {"episode": i, "result": {"stats": {...}}}
      {"fixed": ..., "total_noisy_in_lSet": ...}
    """
    totals = {
        "num_cleaning_calls": 0,
        "fixed": 0,
        "correctly_fixed": 0,
        "clean_corrupted": 0,
        "wrong_to_wrong": 0,
        "total_noisy_in_lSet": 0,
        "skipped": 0,
    }
    if not isinstance(noise_method_results, list):
        return totals

    counter_keys = ["fixed", "correctly_fixed", "clean_corrupted", "wrong_to_wrong", "total_noisy_in_lSet"]

    for raw in noise_method_results:
        if not isinstance(raw, dict):
            continue

        item = raw.get("result", raw)
        if not isinstance(item, dict):
            continue

        status = item.get("status")
        method = item.get("method")
        if status == "skipped" or method == "none":
            totals["skipped"] += 1
            continue

        # New format: counters are directly on result.
        # Some older code: counters are under result["stats"].
        stats = item.get("stats", item)
        if not isinstance(stats, dict):
            stats = {}

        totals["num_cleaning_calls"] += 1
        for key in counter_keys:
            value = stats.get(key, item.get(key, 0))
            try:
                totals[key] += int(value or 0)
            except Exception:
                pass

    return totals



def _sum_npc_filter_results(npc_filter_results: Any) -> Dict[str, Any]:
    """Aggregate NPC detector and noise-dropout statistics across AL queries."""
    totals = {
        "num_filter_calls": 0,
        "pred_noisy_before_dropout": 0,
        "noise_dropout_restored": 0,
        "pred_noisy_after_dropout": 0,
        "oracle_true_noisy": 0,
        "oracle_correctly_flagged_before_dropout": 0,
        "oracle_clean_flagged_before_dropout": 0,
        "oracle_missed_noisy_before_dropout": 0,
        "oracle_correctly_flagged": 0,
        "oracle_clean_flagged": 0,
        "oracle_missed_noisy": 0,
    }
    if not isinstance(npc_filter_results, list):
        return totals

    for item in npc_filter_results:
        if not isinstance(item, dict):
            continue
        totals["num_filter_calls"] += 1
        for key in list(totals.keys()):
            if key == "num_filter_calls":
                continue
            try:
                totals[key] += int(item.get(key, 0) or 0)
            except Exception:
                pass
    return totals

def _safe_sum(xs: Any) -> Optional[float]:
    try:
        return float(sum(float(x) for x in xs))
    except Exception:
        return None


def _light_grid_entry(combo_name: str, pipeline: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    """Compact final-result entry for aggregate JSON."""
    if not isinstance(results, dict):
        return {
            "combo_name": combo_name,
            "status": "failed",
            "reason": "results is not a dict",
        }

    summary = results.get("summary", {}) if isinstance(results.get("summary", {}), dict) else {}
    final_knn = summary.get("final_knn_eval", {}) if isinstance(summary.get("final_knn_eval", {}), dict) else {}

    noise_cfg = pipeline.get("noise", {}) if isinstance(pipeline.get("noise", {}), dict) else {}
    al_cfg = pipeline.get("active_learning", {}) if isinstance(pipeline.get("active_learning", {}), dict) else {}
    nh_cfg = pipeline.get("noise_handling", {}) if isinstance(pipeline.get("noise_handling", {}), dict) else {}
    run_args = pipeline.get("run_args", {}) if isinstance(pipeline.get("run_args", {}), dict) else {}

    loss_fun = run_args.get("loss_fun") or "cross_entropy"

    return _jsonify({
        "combo_name": combo_name,
        "status": results.get("status", "unknown"),
        "reason": results.get("reason"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "exp_dir": results.get("exp_dir"),

        "noise_type": noise_cfg.get("type") or "clean",
        "noise_rate": noise_cfg.get("rate"),
        "al_method": al_cfg.get("method"),
        "nas_filter_method": al_cfg.get("nas_filter_method"),
        "nas_noise_dropout": al_cfg.get("nas_noise_dropout"),
        "nas_noise_dropout_min_fraction": al_cfg.get("nas_noise_dropout_min_fraction"),
        "nas_min_labeled_size": al_cfg.get("nas_min_labeled_size"),
        "nas_aum_max_pred_noisy_rate": al_cfg.get("nas_aum_max_pred_noisy_rate"),
        "noise_method": nh_cfg.get("method") or "none",
        "loss_fun": loss_fun,
        "effective_loss_fun": summary.get("effective_loss_fun"),
        "effective_gce_q": summary.get("effective_gce_q"),
        "effective_bootstrap_beta": summary.get("effective_bootstrap_beta"),
        "seed": run_args.get("seed"),

        "initial_size": run_args.get("initial_size"),
        "budget": al_cfg.get("budget"),
        "max_iter": al_cfg.get("max_iter"),

        "train_size": summary.get("train_size"),
        "test_size": summary.get("test_size"),
        "final_labeled_set_size": summary.get("final_lSet_len"),
        "final_unlabeled_set_size": summary.get("final_uSet_len"),
        "final_val_set_size": summary.get("final_valSet_len"),

        "final_train_acc": summary.get("final_train_acc"),
        "final_val_acc": summary.get("final_val_acc"),
        "final_test_acc": summary.get("final_test_acc"),

        "final_knn_train_acc": final_knn.get("train_acc"),
        "final_knn_val_acc": final_knn.get("val_acc"),
        "final_knn_test_acc": final_knn.get("test_acc"),
        "knn_train_size": final_knn.get("train_size"),
        "knn_val_size": final_knn.get("val_size"),
        "knn_test_size": final_knn.get("test_size"),
        "knn_k_effective": final_knn.get("k_effective"),

        "noise_cleaning_totals": _sum_noise_cleaning_results(summary.get("noise_method_results", [])),
        "npc_filter_totals": _sum_npc_filter_results(summary.get("npc_filter_results", [])),
        "npc_filter_results": summary.get("npc_filter_results", []),

        "total_training_time_s": _safe_sum(summary.get("episode_training_times", [])),
        "total_al_time_s": _safe_sum(summary.get("episode_al_times", [])),
    })


def _grid_json_entry(combo_name: str, pipeline: Dict[str, Any], results: Dict[str, Any], debug_json: bool) -> Dict[str, Any]:
    if debug_json:
        return _jsonify({"pipeline": pipeline, "results": results})
    return _light_grid_entry(combo_name, pipeline, results)


def _load_existing_grid_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        logger.exception("Could not read existing aggregate JSON at %s; starting new object", path)
        return {}


def _write_grid_json(path: str, all_results: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(_jsonify(all_results), f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def main() -> Dict[str, Any]:
    args = parse_args()

    grid_noise = _split_list_arg(args.grid_noise_types) or [args.noise_type]
    grid_al = _split_list_arg(args.grid_al_methods) or [args.al_method]
    grid_noise_methods = _split_list_arg(args.grid_noise_methods) or [args.noise_method]

    grid_out_dir = args.grid_out_dir or os.path.join(os.getcwd(), "grid_runs")
    os.makedirs(grid_out_dir, exist_ok=True)

    # Important:
    # - If --grid_json_path is supplied by the .sh, all Python stages append to the same JSON.
    # - If not supplied, preserve old behavior name inside this stage directory.
    grid_json_path = args.grid_json_path or os.path.join(grid_out_dir, "grid_full_log.json")
    all_results = _load_existing_grid_json(grid_json_path)

    total = len(grid_noise) * len(grid_al) * len(grid_noise_methods)
    for run_idx, (noise_t, al_m, nmethod) in enumerate(itertools.product(grid_noise, grid_al, grid_noise_methods), start=1):
        loss_name = args.loss_fun or "cross_entropy"
        combo_name = (
            f"noise={noise_t or 'clean'}"
            f"__al={al_m or 'cfg'}"
            f"__nm={nmethod or 'none'}"
            f"__loss={loss_name}"
            f"__seed={args.seed or 'NA'}"
        )
        exp_name = args.exp_name if args.exp_name not in [None, "", "auto"] else combo_name
        logger.info("Starting run %d/%d: %s", run_idx, total, combo_name)

        try:
            pipeline = build_pipeline(args, noise_t, al_m, nmethod, exp_name)
            results = run_one(pipeline)
        except Exception as exc:
            logger.exception("Run failed before completion: %s", combo_name)
            pipeline = {"error": "pipeline construction failed"}
            results = {"status": "failed", "reason": repr(exc)}

        # JSON: light by default, full old-style only with --debug_json.
        all_results[combo_name] = _grid_json_entry(combo_name, pipeline, results, bool(args.debug_json))
        _write_grid_json(grid_json_path, all_results)
        logger.info("Updated aggregate JSON at %s", grid_json_path)

        # Keep a text stage log for debugging convenience.
        grid_log_file = os.path.join(grid_out_dir, "grid_full_log.txt")
        with open(grid_log_file, "a") as f:
            f.write("\n" + "=" * 30 + f" GRID RESULT: {combo_name} " + "=" * 30 + "\n")
            f.write(json.dumps(_jsonify(all_results[combo_name]), indent=2, sort_keys=True) + "\n")
            f.write("=" * 80 + "\n")
        logger.info("Updated stage text log at %s", grid_log_file)

    grid_log_file = os.path.join(grid_out_dir, "grid_full_log.txt")
    with open(grid_log_file, "a") as f:
        f.write("\n" + "=" * 30 + " GRID AGGREGATE JSON PATH " + "=" * 30 + "\n")
        f.write(grid_json_path + "\n")
        f.write("=" * 80 + "\n")
    logger.info("Wrote aggregate JSON: %s", grid_json_path)
    return all_results


if __name__ == "__main__":
    main()
