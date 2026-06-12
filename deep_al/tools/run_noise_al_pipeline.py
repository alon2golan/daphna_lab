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
import os
import sys
import json
import logging
import numpy as np
import time
from copy import deepcopy
from typing import Optional

# Ensure repo root is in sys.path for clean imports
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from deep_al.pycls.datasets.noisy_dataset import NoisyDataset
from deep_al.pycls.al.active_learning import ActiveLearning
from deep_al.pycls.core.config import initialize_cfg, normalize_cfg_paths
from deep_al.pycls.datasets.data import Data

logger = logging.getLogger("run_noise_al_pipeline")
logging.basicConfig(level=logging.INFO)


def build_pipeline(noise_type: Optional[str], noise_rate: float, noise_transition_path: Optional[str],
                   al_method: Optional[str], noise_method: Optional[str], cfg_path: Optional[str],
                   seed: Optional[int] = None, budget: Optional[int] = None, initial_size: Optional[int] = None,
                   max_iter: Optional[int] = None, exp_name: Optional[str] = None, features_tag: Optional[str] = None,
                   initial_delta: Optional[float] = None, kernel_type: Optional[str] = None,
                   eval_model_type: Optional[str] = None,
                   max_epoch: Optional[int] = None):
    """Builds a pipeline dict describing the requested experiment.

    This function is intentionally lightweight and refrains from executing heavy
    training code. It wires together available components and shows where to
    plug dataset loading, AL loop and noise-handling methods.
    """

    pipeline = {
        "cfg": cfg_path,
        "noise": {
            "type": noise_type,
            "rate": noise_rate,
            "transition_matrix": noise_transition_path,
        },
        "active_learning": {
            "method": al_method,
            "budget": budget,
            "max_iter": max_iter,
        },
        "noise_handling": {
            "method": noise_method,
        },
        "run_args": {
            "seed": seed,
            "initial_size": initial_size,
            "exp_name": exp_name,
            "features_tag": features_tag,
            "initial_delta": initial_delta,
            "kernel_type": kernel_type,
            "eval_model_type": eval_model_type,
            "max_epoch": max_epoch,
        }
    }

    logger.info("Pipeline built:")
    logger.info(json.dumps(pipeline, indent=2))

    return pipeline


def example_run(pipeline: dict):
    """Wired example run that uses the repository's config and dataset loaders.

    This function performs light-weight wiring only:
      - loads the config using initialize_cfg
      - prepares datasets via Data.getDataset
      - creates L/U/V splits via Data.makeLUVSets
      - wraps the training dataset with NoisyDataset (if requested)
      - instantiates the ActiveLearning orchestrator

    It does NOT run training or full AL rounds; those should be implemented where noted.
    """

    # 1) Build config
    cfg_file = pipeline.get("cfg")

    # Build args_dict from pipeline run_args and active_learning entries
    args_dict = {}
    run_args = pipeline.get('run_args', {})
    if run_args.get('seed') is not None:
        args_dict['RNG_SEED'] = int(run_args.get('seed'))
    if pipeline.get('active_learning', {}).get('budget') is not None:
        args_dict['ACTIVE_LEARNING.BUDGET_SIZE'] = int(pipeline['active_learning'].get('budget'))
    if pipeline.get('active_learning', {}).get('max_iter') is not None:
        args_dict['ACTIVE_LEARNING.MAX_ITER'] = int(pipeline['active_learning'].get('max_iter'))
    if run_args.get('exp_name') is not None:
        args_dict['EXP_NAME'] = run_args.get('exp_name')
    if run_args.get('features_tag') is not None:
        args_dict['DATASET.FEATURES_TAG'] = run_args.get('features_tag')
    if run_args.get('initial_delta') is not None:
        args_dict['ACTIVE_LEARNING.INITIAL_DELTA'] = float(run_args.get('initial_delta'))
    if run_args.get('kernel_type') is not None:
        args_dict['KERNEL_TYPE'] = run_args.get('kernel_type')
    if run_args.get('eval_model_type') is not None:
        args_dict['EVAL_MODEL_TYPE'] = run_args.get('eval_model_type')
    if run_args.get('max_epoch') is not None:
        args_dict['OPTIM.MAX_EPOCH'] = int(run_args.get('max_epoch'))

    cfg = initialize_cfg(cfg_file, args_dict)
    cfg = normalize_cfg_paths(cfg)

    # Ensure some optional config keys exist with safe defaults (avoid AttributeError in downstream code)
    try:
        _ = cfg.MODEL.LINEAR_FROM_FEATURES
    except Exception:
        cfg.MODEL.LINEAR_FROM_FEATURES = False
    try:
        _ = cfg.MODEL.LINEAR_FROM_IMAGES
    except Exception:
        cfg.MODEL.LINEAR_FROM_IMAGES = False
    try:
        _ = cfg.MODEL.USE_1NN
    except Exception:
        cfg.MODEL.USE_1NN = False

    # Ensure output directories
    if not cfg.OUT_DIR:
        cfg.OUT_DIR = os.path.join(os.getcwd(), "output")
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    cfg.EXP_DIR = os.path.join(cfg.OUT_DIR, "onboarding_runs")
    os.makedirs(cfg.EXP_DIR, exist_ok=True)

    # Seed default
    if cfg.RNG_SEED is None:
        cfg.RNG_SEED = np.random.randint(100)

    logger.info("Loaded config from %s. EXP_DIR=%s", cfg_file, cfg.EXP_DIR)

    # 2) Load datasets using Data helper
    data_obj = Data(cfg)
    train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True)
    test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True)

    logger.info("Train size: %s, Test size: %s", train_size, test_size)

    # 3) Determine initial labeled ratio from initial_size if provided, then create L/U/V splits
    initial_size = pipeline.get('run_args', {}).get('initial_size')
    try:
        initial_size = int(initial_size)
        cfg.ACTIVE_LEARNING.INIT_L_RATIO = float(initial_size) / float(train_size)
        logger.info("Setting ACTIVE_LEARNING.INIT_L_RATIO to %s (initial_size=%d, train_size=%d)",
                    cfg.ACTIVE_LEARNING.INIT_L_RATIO, initial_size, train_size)
    except Exception:
        logger.exception("Failed to set initial labeled ratio from initial_size; using cfg default %s",
                         cfg.ACTIVE_LEARNING.INIT_L_RATIO)

    lSet_path, uSet_path, valSet_path = data_obj.makeLUVSets(
        train_split_ratio=cfg.ACTIVE_LEARNING.INIT_L_RATIO,
        val_split_ratio=cfg.DATASET.VAL_RATIO,
        data=train_data,
        seed_id=cfg.RNG_SEED,
        save_dir=cfg.EXP_DIR,
    )

    lSet, uSet, valSet = data_obj.loadPartitions(lSetPath=lSet_path, uSetPath=uSet_path, valSetPath=valSet_path)

    # 4) Apply noise wrapper to the training dataset if requested
    noisy_applied = False
    transition = None
    if pipeline.get("noise", {}).get("type") and pipeline.get("noise", {}).get("type") != "clean":
        if NoisyDataset is None:
            logger.warning("NoisyDataset not available; skipping noise injection")
        else:
            if pipeline["noise"].get("transition_matrix"):
                try:
                    transition = np.load(pipeline["noise"]["transition_matrix"])
                except Exception:
                    logger.exception("Failed to load transition matrix; ignoring it")
            logger.info("Applying noise wrapper: type=%s rate=%s", pipeline["noise"]["type"], pipeline["noise"]["rate"])
            train_data = NoisyDataset(train_data,
                                      noise_type=pipeline["noise"]["type"],
                                      noise_rate=pipeline["noise"]["rate"],
                                      num_classes=getattr(cfg.MODEL, 'NUM_CLASSES', None),
                                      seed=cfg.RNG_SEED,
                                      noise_transition_matrix=transition)
            noisy_applied = True

    # 5) Prepare train labels for AL constructor (use clean labels when available)
    if hasattr(train_data, 'clean_labels'):
        train_labels = train_data.clean_labels
    elif hasattr(train_data, 'targets'):
        train_labels = np.array(getattr(train_data, 'targets'))
    else:
        train_labels = None

    # 6) Set AL method in cfg if requested by pipeline
    if pipeline.get('active_learning', {}).get('method'):
        cfg.ACTIVE_LEARNING.SAMPLING_FN = pipeline['active_learning']['method']

    # 7) Instantiate ActiveLearning orchestrator
    al_obj = None
    if ActiveLearning is None:
        logger.warning("ActiveLearning orchestrator not available in this environment.")
    else:
        try:
            al_obj = ActiveLearning(data_obj, cfg, train_labels, lSet)
            logger.info("ActiveLearning instantiated with sampling_fn=%s", cfg.ACTIVE_LEARNING.SAMPLING_FN)
        except Exception:
            logger.exception("Failed to instantiate ActiveLearning")

    # 8) Run Active Learning episodes (training + sampling)
    # Import train_al utilities for training/eval loop
    try:
        from deep_al.tools import train_al as train_al_module
        PlottingManager = train_al_module.PlottingManager
    except Exception:
        train_al_module = None
        PlottingManager = None

    if train_al_module is None:
        logger.warning("train_al utilities not available; returning wiring summary.")
        summary = {
            'train_size': int(train_size),
            'test_size': int(test_size),
            'lSet_len': int(len(lSet)),
            'uSet_len': int(len(uSet)),
            'valSet_len': int(len(valSet)),
            'noisy_wrapper_applied': bool(noisy_applied),
            'al_obj_created': bool(al_obj is not None),
        }
        return {'status': 'wired', 'summary': summary}

    # small helper: label-cleaning stub
    def run_label_cleaning(data_obj, train_data, lSet, cfg, checkpoint_file=None, threshold=0.9, max_fix=None):
        """Simple label-cleaning stub.

        - Loads model from checkpoint_file (if provided)
        - Predicts labels for current labeled set and replaces labels where model is confident
        """
        if checkpoint_file is None:
            logger.info("No checkpoint provided - skipping label_cleaning.")
            return {'fixed': 0, 'changed_indices': []}

        try:
            model = train_al_module.mh.get_model(cfg, checkpoint_file)
            device = 'cuda' if (cfg.NUM_GPUS > 0 and __import__('torch').cuda.is_available()) else 'cpu'
            model.to(device)
            model.eval()
        except Exception:
            logger.exception("Failed to load model for cleaning; skipping")
            return {'fixed': 0, 'changed_indices': []}

        # Process labeled indices in deterministic batches (don't rely on DataLoader sampling order)
        import torch
        fixed = 0
        changed = []
        batch_size = cfg.TRAIN.BATCH_SIZE
        with torch.no_grad():
            for start in range(0, len(lSet), batch_size):
                batch_indices = lSet[start:start + batch_size]
                inputs_list = []
                labels_list = []
                for idx in batch_indices:
                    sample, lbl = train_data[int(idx)]
                    inputs_list.append(sample)
                    labels_list.append(lbl)
                # Stack inputs into tensor batch
                try:
                    inputs = torch.stack(
                        [inp if isinstance(inp, torch.Tensor) else torch.tensor(inp) for inp in inputs_list]).to(device)
                except Exception:
                    # fallback: convert to float tensor
                    inputs = torch.tensor(np.stack([np.array(inp) for inp in inputs_list])).to(device)
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confs, preds = probs.max(dim=1)
                confs = confs.cpu().numpy()
                preds = preds.cpu().numpy()

                for i in range(len(batch_indices)):
                    cur_idx = int(batch_indices[i])
                    pred = int(preds[i])
                    conf = float(confs[i])
                    cur_label = None
                    # read current label from dataset (support NoisyDataset and common attr targets)
                    if hasattr(train_data, 'noisy_labels'):
                        cur_label = int(train_data.noisy_labels[cur_idx])
                    elif hasattr(train_data, 'targets'):
                        cur_label = int(train_data.targets[cur_idx])
                    else:
                        continue

                    if pred != cur_label and conf >= threshold:
                        # fix label in-place
                        if hasattr(train_data, 'noisy_labels'):
                            train_data.noisy_labels[cur_idx] = pred
                            if hasattr(train_data, 'clean_labels'):
                                train_data.is_noisy[cur_idx] = (train_data.clean_labels[cur_idx] != pred)
                        elif hasattr(train_data, 'targets'):
                            # update targets stored in dataset; many dataset implementations expect list
                            try:
                                train_data.targets[cur_idx] = pred
                            except Exception:
                                # if list, convert
                                tlist = list(train_data.targets)
                                tlist[cur_idx] = pred
                                train_data.targets = tlist
                        fixed += 1
                        changed.append(cur_idx)
                        if max_fix and fixed >= max_fix:
                            break
                if max_fix and fixed >= max_fix:
                    break
        logger.info("Label cleaning: fixed %d labels", fixed)
        return {'fixed': fixed, 'changed_indices': changed}

    # Prepare dataloaders
    lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    test_loader = data_obj.getTestLoader(data=test_data, test_batch_size=cfg.TRAIN.BATCH_SIZE, seed_id=cfg.RNG_SEED)
    if valSet_loader is None:
        valSet_loader = lSet_loader

    # Initialize model, optimizer and plotting
    model = train_al_module.mh.get_model(cfg)
    optimizer = train_al_module.optim.construct_optimizer(cfg, model)
    opt_init_state = deepcopy(optimizer.state_dict()) if not getattr(cfg.MODEL, 'USE_1NN', False) else None
    model_init_state = deepcopy(model.state_dict().copy()) if not getattr(cfg.MODEL, 'USE_1NN', False) else None

    plotting_manager = PlottingManager()

    # Keep simple timing/metric trackers
    episode_training_times = []
    episode_al_times = []
    all_epoch_times = []
    final_train_acc = final_val_acc = final_test_acc = 0.0

    # Demo combo defaults: if pipeline entries empty, set demo (sym + typiclust + label_cleaning)
    if pipeline.get('noise', {}).get('type') is None and pipeline.get('active_learning', {}).get(
            'method') is None and pipeline.get('noise_handling', {}).get('method') is None:
        logger.info("No selections provided - using demo combo: sym + typiclust + label_cleaning")
        pipeline['noise']['type'] = 'sym'
        pipeline['noise']['rate'] = 0.2
        cfg.ACTIVE_LEARNING.SAMPLING_FN = 'typiclust'
        pipeline['active_learning']['method'] = 'typiclust'
        pipeline['noise_handling']['method'] = 'label_cleaning'

    # Run episodes
    checkpoint_file = None
    for cur_episode in range(0, cfg.ACTIVE_LEARNING.MAX_ITER + 1):
        logger.info("======== EPISODE %d BEGINS ========", cur_episode)
        episode_dir = os.path.join(cfg.EXP_DIR, f'episode_{cur_episode}')
        os.makedirs(episode_dir, exist_ok=True)
        cfg.EPISODE_DIR = episode_dir

        # Train and evaluate
        try:
            checkpoint_file, episode_train_time, epoch_times, best_train_acc, best_val_acc, test_acc = \
                train_al_module.train_and_eval_model(cfg, cur_episode, lSet_loader, model, optimizer, test_loader,
                                                     valSet_loader, plotting_manager)
        except Exception:
            logger.exception("train_and_eval_model failed; aborting AL loop")
            return {'status': 'failed', 'reason': 'train_and_eval_model failure'}

        episode_training_times.append(episode_train_time)
        all_epoch_times.extend(epoch_times or [])
        final_train_acc = best_train_acc
        final_val_acc = best_val_acc
        final_test_acc = test_acc

        # If last episode, save sets and finish
        if cur_episode == cfg.ACTIVE_LEARNING.MAX_ITER:
            data_obj.saveSet(lSet, 'lSet', cfg.EPISODE_DIR)
            data_obj.saveSet(uSet, 'uSet', cfg.EPISODE_DIR)
            break

        # Active sampling
        al_start_time = time.time()
        try:
            lSet, lSet_loader, uSet, valSet_loader = train_al_module.active_sampling_part(cfg, checkpoint_file,
                                                                                          cur_episode,
                                                                                          data_obj, lSet, lSet_loader,
                                                                                          train_data, uSet, valSet,
                                                                                          valSet_loader, al_obj)
        except Exception:
            logger.exception("active_sampling_part failed; aborting AL loop")
            return {'status': 'failed', 'reason': 'active_sampling_part failure'}
        al_time = time.time() - al_start_time
        episode_al_times.append(al_time)

        # Apply noise-handling method if requested (label_cleaning demo)
        if pipeline.get('noise_handling', {}).get('method') == 'label_cleaning':
            logger.info("Running label-cleaning on labeled set after sampling (threshold=0.9)")
            try:
                res = run_label_cleaning(data_obj, train_data, lSet, cfg, checkpoint_file=checkpoint_file,
                                         threshold=0.9)
                logger.info("Label cleaning result: %s", res)
            except Exception:
                logger.exception("Label cleaning failed; continuing")

        # Reinitialize model if not fine-tuning
        if not cfg.ACTIVE_LEARNING.FINE_TUNE:
            logger.info('Starting model from scratch for next episode')
            model = train_al_module.mh.get_model(cfg, model_init_state=model_init_state)
            optimizer = train_al_module.optim.construct_optimizer(cfg, model, opt_init_state=opt_init_state)

        # Remove checkpoint file
        if checkpoint_file is not None:
            try:
                os.remove(checkpoint_file)
            except Exception:
                pass

    # Summary
    summary = {
        'train_size': int(train_size),
        'test_size': int(test_size),
        'final_lSet_len': int(len(lSet)),
        'final_uSet_len': int(len(uSet)),
        'final_valSet_len': int(len(valSet)),
        'noisy_wrapper_applied': bool(noisy_applied),
        'al_obj_created': bool(al_obj is not None),
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc,
        'final_test_acc': final_test_acc,
        'total_episode_training_times': episode_training_times,
        'total_episode_al_times': episode_al_times,
    }

    return {'status': 'completed', 'summary': summary}


def parse_args():
    parser = argparse.ArgumentParser(description="Run noise x AL x noise-handling pipeline (skeleton)")

    # Core experiment config (reuse existing cfg format if desired)
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', required=True, type=str)

    # Noise specification
    parser.add_argument('--noise_type', help='Noise type (e.g. sym, asym, clean)', default=None, type=str)
    parser.add_argument('--noise_rate', help='Noise rate (0.0-1.0)', default=0.0, type=float)
    parser.add_argument('--noise_transition', help='Path to noise transition matrix (npz/npy)', default=None, type=str)

    # Active learning method
    parser.add_argument('--al_method', help='Active learning method (e.g. typiclust, probcover, misp)', default=None,
                        type=str)
    parser.add_argument('--budget', help='Budget per AL round', default=None, type=int)
    parser.add_argument('--max_iter', help='Max AL iterations', default=None, type=int)

    # Noise handling method
    parser.add_argument('--noise_method', help='Noise handling method (e.g. cleaning, robust_loss)', default=None,
                        type=str)

    # Additional run parameters
    parser.add_argument('--seed', help='Random seed', default=None, type=int)
    parser.add_argument('--initial_size', help='Initial labeled set size', default=None, type=int)
    parser.add_argument('--exp_name', help='Experiment name', default=None, type=str)
    parser.add_argument('--features_tag', help='Features tag', default=None, type=str)
    parser.add_argument('--initial_delta', help='Initial delta (ProbCover/DCoM)', default=None, type=float)
    parser.add_argument('--kernel_type', help='Kernel type (rbf/tophat)', default=None, type=str)
    parser.add_argument('--eval_model_type', help='Eval model type alias', default=None, type=str)
    parser.add_argument('--max_epoch', help='Max training epochs per AL round', default=None, type=int)

    parser.add_argument('--out', help='Output results path (json)', default=None, type=str)

    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = build_pipeline(
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        noise_transition_path=args.noise_transition,
        al_method=args.al_method,
        noise_method=args.noise_method,
        cfg_path=args.cfg_file,
        seed=args.seed,
        budget=args.budget,
        initial_size=args.initial_size,
        max_iter=args.max_iter,
        exp_name=args.exp_name,
        features_tag=args.features_tag,
        initial_delta=args.initial_delta,
        kernel_type=args.kernel_type,
        eval_model_type=args.eval_model_type,
        max_epoch=args.max_epoch,
    )

    results = example_run(pipeline)

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info("Wrote results to %s", args.out)
    else:
        logger.info("Run finished. Results: %s", results)


if __name__ == "__main__":
    main()
