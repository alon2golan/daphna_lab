import os
import sys
from datetime import datetime
import argparse
import numpy as np
import time

# Ensure repo root is in sys.path for clean imports
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
from copy import deepcopy

from deep_al.pycls.al.active_learning import ActiveLearning
from deep_al.pycls.al.distillation import prepare_distillation_targets
from deep_al.pycls.al.propagation import propagate_labels, build_al_resample_fn
from deep_al.pycls.core.config import cfg, dump_cfg, setup_cfg
import deep_al.pycls.core.losses as losses
import deep_al.pycls.core.optimizer as optim
from deep_al.pycls.datasets.data import Data
import deep_al.pycls.utils.checkpoint as cu
import deep_al.pycls.utils.logging as lu
import deep_al.pycls.utils.metrics as mu
import deep_al.pycls.utils.net as nu
import deep_al.pycls.utils.model_handler as mh
from deep_al.pycls.utils.meters import TestMeter
from deep_al.pycls.utils.meters import TrainMeter
from deep_al.pycls.utils.meters import ValMeter
from deep_al.pycls.utils.plotting_state import PlottingManager

logger = lu.get_logger(__name__)


def argparser():
    parser = argparse.ArgumentParser(description='Active Learning - Image Classification')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', required=True, type=str)
    parser.add_argument('--exp-name', help='Experiment Name', required=True, type=str)
    parser.add_argument('--al', help='AL Method', required=True, type=str) ## active learning type -> write TypiClust_rp\_dc
    parser.add_argument('--budget', help='Budget Per Round', required=True, type=int) ## in each iteration how much labels can I get
    parser.add_argument('--initial_size', help='Size of the initial random labeled set', default=0, type=int)
    parser.add_argument('--seed', help='Random seed', default=1, type=int)
    parser.add_argument('--eval_model_type', help='eval_model_type', type=str) # train the eval model from features not images
    parser.add_argument('--features_tag', help='Tag to specify which feature representation to use (e.g., "alon", "itai")', type=str, default=None)
    parser.add_argument('--initial_delta', help='Relevant only for ProbCover and DCoM', default=-1, type=float)
    parser.add_argument('--kernel_type', default='rbf', type=str)
    parser.add_argument('--diff_method', default='abs_diff', type=str)
    parser.add_argument('--confidence_method', default='margin', type=str)
    parser.add_argument('--cont_method', default='positive', type=str)
    parser.add_argument('--k_logistic', default=50, type=int)
    parser.add_argument('--a_logistic', default=0.8, type=float)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--soft_border_val', default=0.5, type=float)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--high_budget', action='store_true')
    parser.add_argument('--norm_importance', action='store_true')
    parser.add_argument('--sparse_ds', action='store_true')
    parser.add_argument('--decrease_alpha', action='store_true')
    parser.add_argument('--max_iter', default=32, type=int)
    parser.add_argument('--initial_sigma', default=1.0, type=float)
    parser.add_argument('--sparse_K', action='store_true')
    parser.add_argument('--features_normalized', default=True, action=argparse.BooleanOptionalAction,
                        help='Whether to L2-normalize dataset features (default: True)')
    parser.add_argument('--matrix_dtype', type=str, default=None, choices=['float32', 'float16'], help="Choose torch dtype: float32 or float16")
    parser.add_argument('--max_epoch', default=None, type=int, help='Max training epochs per AL round')
    parser.add_argument('--base_lr', default=None, type=float, help='Override base learning rate')
    parser.add_argument('--use_mlp', action='store_true',
                        help='Use MLP head instead of linear probe for from_features evaluation')
    parser.add_argument('--mlp_hidden', default=None, type=int, help='MLP hidden layer size (default 256)')
    parser.add_argument('--mlp_dropout', default=None, type=float, help='MLP dropout rate')
    parser.add_argument('--calc_method', default='max', type=str,
                        help='BAYES_MISP scoring rule: max, entropy, margin, or margin_not_normalized')
    parser.add_argument('--distribution_cont_weight_method', default='weighted', type=str,
                        choices=['weighted', 'equal'], help='BAYES_MISP per-point weighting for the contribution sum')
    parser.add_argument('--c_normalization', default='sum', type=str, choices=['sum', 'softmax'],
                        help='BAYES_MISP normalization applied to the C matrix before weighting')
    parser.add_argument('--k_sparsity_threshold', default=0.0, type=float,
                        help='BAYES_MISP minimum kernel value kept when SPARSE_K is set')
    parser.add_argument('--local_alpha', action='store_true',
                        help='BAYES_MISP: init C from per-point priors instead of a constant alpha')
    parser.add_argument('--local_alpha_oracle_method', default='entropy', type=str, choices=['entropy', 'max'])
    parser.add_argument('--alpha_lower_bound', default=0.01, type=float)
    parser.add_argument('--alpha_upper_bound', default=1.0, type=float)
    parser.add_argument('--kernel_build_batch_size', default=1024, type=int)
    parser.add_argument('--class_weighting_method', default='none', type=str,
                        choices=['none', 'linear', 'square', 'exp'])

    # --- Distillation / propagation (see pycls.al.distillation / pycls.al.propagation) ---
    parser.add_argument('--distillation_training', action='store_true',
                        help='Enable soft-target distillation loss alongside CE during training '
                             '(requires the AL algorithm to expose C_general, e.g. bayes_misp)')
    parser.add_argument('--distillation_threshold', default=0.5, type=float,
                        help='Minimum normalized max-probability for a point to participate in distillation')
    parser.add_argument('--distillation_temperature', default=3.0, type=float,
                        help='Temperature for the distillation KL loss')
    parser.add_argument('--distill_factor', default=1.0, type=float,
                        help='Scalar weight applied to the distillation loss term before adding it to CE')
    parser.add_argument('--ck_c_normalization', default='sum', type=str, choices=['sum', 'softmax'],
                        help='Normalization applied to C_general when preparing distillation soft targets')
    parser.add_argument('--propagate_c_labels', action='store_true',
                        help='Apply label propagation to C_general before preparing distillation targets '
                             '(requires --distillation_training)')
    parser.add_argument('--propagation_iterations', default=1, type=int)
    parser.add_argument('--propagation_sender_mode', default='all', type=str,
                        choices=['all', 'high_confidence', 'al_constant', 'al_updated'])
    parser.add_argument('--propagation_sender_threshold', default=0.5, type=float)
    parser.add_argument('--propagation_sender_budget', default=None, type=int)
    parser.add_argument('--propagation_receiver_mode', default='all', type=str,
                        choices=['all', 'high_confidence', 'al_constant', 'al_updated'])
    parser.add_argument('--propagation_receiver_threshold', default=None, type=float)
    parser.add_argument('--propagation_receiver_budget', default=None, type=int)
    return parser


def build_distillation_targets(cfg, al_obj, train_labels, lSet, uSet):
    """
    Build soft targets/mask/weights for distillation from the AL algorithm's
    evidence matrix, optionally propagating labels first.

    Requires the AL algorithm (al_obj.al_algorithm) to expose C_general and
    labeled_points_mask_general (BAYES_MISP and CoverageMatrixMethod both do).
    Returns None if the algorithm doesn't expose what's needed, so callers can
    skip distillation gracefully instead of crashing on unsupported AL methods.
    """
    al_algorithm = al_obj.al_algorithm
    if not hasattr(al_algorithm, 'C_general') or not hasattr(al_algorithm, 'labeled_points_mask_general'):
        print(f'[Distillation] Warning: {type(al_algorithm).__name__} does not expose '
              f'C_general/labeled_points_mask_general — skipping distillation this episode')
        return None

    C_matrix = al_algorithm.C_general
    labeled_mask = al_algorithm.labeled_points_mask_general

    if getattr(cfg, 'PROPAGATE_C_LABELS', False):
        K_matrix = getattr(al_algorithm, 'K_general', None)
        if K_matrix is None:
            print('[Distillation] Warning: PROPAGATE_C_LABELS set but algorithm has no K_general — skipping propagation')
        else:
            labeled_mask_np = np.asarray(
                labeled_mask.cpu() if torch.is_tensor(labeled_mask) else labeled_mask, dtype=bool)
            source_indices = np.where(labeled_mask_np)[0]
            source_labels = np.asarray(train_labels)[source_indices]

            sender_mode = getattr(cfg, 'PROPAGATION_SENDER_MODE', 'all')
            receiver_mode = getattr(cfg, 'PROPAGATION_RECEIVER_MODE', 'all')
            sender_budget = getattr(cfg, 'PROPAGATION_SENDER_BUDGET', None)
            receiver_budget = getattr(cfg, 'PROPAGATION_RECEIVER_BUDGET', None)

            al_resample_fn = None
            if sender_mode == 'al_updated' or receiver_mode == 'al_updated':
                al_resample_fn = build_al_resample_fn(al_algorithm, cfg, lSet, uSet, budget=sender_budget)

            print(f'[Distillation] Applying label propagation: '
                  f'{getattr(cfg, "PROPAGATION_ITERATIONS", 1)} iterations, '
                  f'sender_mode={sender_mode!r}, receiver_mode={receiver_mode!r}')
            C_matrix = propagate_labels(
                C_matrix, K_matrix, labeled_mask, source_indices, source_labels,
                alpha=getattr(al_algorithm, 'alpha', getattr(cfg, 'ALPHA', 0.5)),
                num_iterations=getattr(cfg, 'PROPAGATION_ITERATIONS', 1),
                sender_mode=sender_mode,
                sender_threshold=getattr(cfg, 'PROPAGATION_SENDER_THRESHOLD', 0.5),
                sender_budget=sender_budget,
                receiver_mode=receiver_mode,
                receiver_threshold=getattr(cfg, 'PROPAGATION_RECEIVER_THRESHOLD', None),
                receiver_budget=receiver_budget,
                al_resample_fn=al_resample_fn,
                receiver_al_resample_fn=al_resample_fn,
                normalization=getattr(cfg, 'CK_C_NORMALIZATION', 'sum'),
            )
            print('[Distillation] Label propagation complete.')

    soft_targets, distill_mask, distill_weights = prepare_distillation_targets(
        C_matrix, threshold=getattr(cfg, 'DISTILLATION_THRESHOLD', 0.5),
        labeled_mask=labeled_mask, train_labels=train_labels,
        normalization=getattr(cfg, 'CK_C_NORMALIZATION', 'sum'),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return soft_targets.to(device), distill_mask.to(device), distill_weights.to(device)


def attach_distillation_targets(cfg, al_obj, train_labels, lSet, uSet, lSet_loader):
    """
    Compute distillation targets from the current AL algorithm state and attach
    them to lSet_loader.dataset, where train_epoch looks them up per batch by
    the global sample index (see IndexedDatasetWrapper / getIndexesDataLoader's
    return_index=True). No-op (with a warning) if the AL algorithm doesn't
    expose what's needed.
    """
    if lSet_loader is None:
        return
    targets = build_distillation_targets(cfg, al_obj, train_labels, lSet, uSet)
    if targets is None:
        lSet_loader.dataset.soft_targets = None
        lSet_loader.dataset.distill_mask = None
        lSet_loader.dataset.distill_weights = None
        return
    soft_targets, distill_mask, distill_weights = targets
    lSet_loader.dataset.soft_targets = soft_targets
    lSet_loader.dataset.distill_mask = distill_mask
    lSet_loader.dataset.distill_weights = distill_weights
    al_algorithm = al_obj.al_algorithm
    lSet_loader.dataset.labeled_mask = getattr(al_algorithm, 'labeled_points_mask_general', None)


def is_eval_epoch(cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (
        (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0 or
        (cur_epoch + 1) == cfg.OPTIM.MAX_EPOCH
    )


def prepare_datasets(cfg, args):
    """Prepare and load datasets including sparse dataset handling.
    
    Args:
        cfg: Configuration object
        args: Command line arguments
        
    Returns:
        Tuple of (data_obj, train_data, test_data, train_size, test_size)
    """
    data_obj = Data(cfg)
    train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True)
    test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True)

    if cfg.SPARSE_DS:
        train_spare_inds = np.sort(np.load(f"/cs/labs/daphna/itai.david/py_repos/TypiClust/results/t-sne/{cfg['DATASET']['NAME']}_train_sparse_indices.npy"))
        test_spare_inds = np.sort(np.load(f"/cs/labs/daphna/itai.david/py_repos/TypiClust/results/t-sne/{cfg['DATASET']['NAME']}_test_sparse_indices.npy"))

        train_data.features = train_data.features[train_spare_inds.astype(int)]
        train_data.targets = np.array(train_data.targets)[train_spare_inds].tolist()
        train_data.data = train_data.data[train_spare_inds.astype(int)]

        test_data.features = test_data.features[test_spare_inds]
        test_data.targets = np.array(test_data.targets)[test_spare_inds].tolist()
        test_data.data = test_data.data[test_spare_inds.astype(int)]

    return data_obj, train_data, test_data, train_size, test_size


def setup_experiment_directory(cfg):
    """Setup experiment directory structure and configuration.
    
    Args:
        cfg: Configuration object
        
    Returns:
        exp_dir: Path to the experiment directory
    """
    # Get the output directory ready (paths are already normalized from config.py)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    # Create "DATASET/MODEL TYPE" specific directory
    dataset_out_dir = os.path.join(cfg.OUT_DIR, cfg.DATASET.NAME, cfg.MODEL.TYPE)
    os.makedirs(dataset_out_dir, exist_ok=True)

    # Create day-specific directory
    now = datetime.now()
    day_dir = f'{now.year}_{now.month}_{now.day}'
    full_day_dir = os.path.join(dataset_out_dir, day_dir)
    os.makedirs(full_day_dir, exist_ok=True)
    
    # Generate experiment name with prefix
    if cfg.EXP_NAME == 'auto':
        exp_prefix = ""
    else:
        exp_prefix = f"{cfg.EXP_NAME}_"
    
    exp_suffix = f'{now.year}_{now.month}_{now.day}_{now.hour:02}{now.minute:02}{now.second:02}_{now.microsecond}'
    exp_name = f'{exp_prefix}{cfg.DATASET.NAME}_{cfg.ACTIVE_LEARNING.SAMPLING_FN}_{cfg.EVAL_MODEL_TYPE}_{exp_suffix}'
    
    # Create experiment directory
    exp_dir = os.path.join(full_day_dir, exp_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
        print("Experiment Directory is {}.\n".format(exp_dir))
    else:
        print("Experiment Directory Already Exists: {}. Reusing it may lead to loss of old logs in the directory.\n".format(exp_dir))
    
    cfg.EXP_DIR = exp_dir
    return exp_dir


def main(cfg):
    plotting_manager = PlottingManager()
    print("RNG_SEED is set to {}".format(cfg.RNG_SEED))
    # Setting up GPU args
    use_cuda = (cfg.NUM_GPUS > 0) and torch.cuda.is_available()

    # Auto assign a RNG_SEED when not supplied a value
    if cfg.RNG_SEED is None:
        cfg.RNG_SEED = np.random.randint(100)

    # Setup experiment directory structure
    setup_experiment_directory(cfg)

    # Save the config file in EXP_DIR and OUT_DIR
    dump_cfg(cfg)

    # Setup Logger
    lu.setup_logging(cfg)

    # Dataset preparing steps
    print("\n======== PREPARING DATA AND MODEL ========\n")
    data_obj, train_data, test_data, train_size, test_size = prepare_datasets(cfg, args)

    cfg.ACTIVE_LEARNING.INIT_L_RATIO = args.initial_size / train_size
    print("\nDataset {} Loaded Sucessfully.\nTotal Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))
    logger.info("Dataset {} Loaded Sucessfully. Total Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))

    lSet_path, uSet_path, valSet_path = data_obj.makeLUVSets(train_split_ratio=cfg.ACTIVE_LEARNING.INIT_L_RATIO, \
        val_split_ratio=cfg.DATASET.VAL_RATIO, data=train_data, seed_id=cfg.RNG_SEED, save_dir=cfg.EXP_DIR)

    cfg.ACTIVE_LEARNING.LSET_PATH = lSet_path
    cfg.ACTIVE_LEARNING.USET_PATH = uSet_path
    cfg.ACTIVE_LEARNING.VALSET_PATH = valSet_path

    lSet, uSet, valSet = data_obj.loadPartitions(lSetPath=cfg.ACTIVE_LEARNING.LSET_PATH, \
            uSetPath=cfg.ACTIVE_LEARNING.USET_PATH, valSetPath = cfg.ACTIVE_LEARNING.VALSET_PATH)
    model = mh.get_model(cfg)
    if cfg.HIGH_BUDGET:
        seed = 42
        rng = np.random.default_rng(seed)
        lset_indices_from_uset = rng.choice(uSet.size, size=10000, replace=False)
        lset_mask = np.zeros_like(uSet, dtype=bool)
        lset_mask[lset_indices_from_uset] = True
        lSet = uSet[lset_mask]
        uSet = uSet[~lset_mask]

    train_labels = train_data.targets
    al_obj = ActiveLearning(data_obj, cfg, train_labels, lSet, plotting_manager)
    
    if len(lSet) == 0:
        print('Labeled Set is Empty - Sampling an Initial Pool')
        activeSet, new_uSet = al_obj.sample_from_uSet(model, lSet, uSet, train_data)
        # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
        lSet = np.append(lSet, activeSet)
        uSet = new_uSet

    print("Data Partitioning Complete. \nLabeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))
    logger.info("Labeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))

    distillation_enabled = getattr(cfg, 'DISTILLATION_TRAINING', False)

    # Preparing dataloaders for initial training
    lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data,
                                                return_index=distillation_enabled)
    valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    test_loader = data_obj.getTestLoader(data=test_data, test_batch_size=cfg.TRAIN.BATCH_SIZE, seed_id=cfg.RNG_SEED)

    if valSet_loader is None:
        valSet_loader = lSet_loader

    if distillation_enabled:
        attach_distillation_targets(cfg, al_obj, train_labels, lSet, uSet, lSet_loader)

    # Initialize the model.  
    print("model: {}\n".format(cfg.MODEL.TYPE))
    logger.info("model: {}\n".format(cfg.MODEL.TYPE))

    # Construct the optimizer
    model = mh.get_model(cfg)
    optimizer = optim.construct_optimizer(cfg, model)
    opt_init_state = deepcopy(optimizer.state_dict()) if not cfg.MODEL.USE_1NN else None
    model_init_state = deepcopy(model.state_dict().copy()) if not cfg.MODEL.USE_1NN else None

    # print("optimizer: {}\n".format(optimizer))
    # logger.info("optimizer: {}\n".format(optimizer))

    print("AL Query Method: {}\nMax AL Episodes: {}\n".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))
    logger.info("AL Query Method: {}\nMax AL Episodes: {}\n".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))

    # Initialize timing trackers
    all_epoch_times = []  # All epoch times across all episodes
    episode_training_times = []  # Total training time per episode
    episode_al_times = []  # AL sampling time per episode
    
    # Initialize accuracy trackers for final episode
    final_train_acc = 0.0
    final_val_acc = 0.0
    final_test_acc = 0.0

    for cur_episode in range(0, cfg.ACTIVE_LEARNING.MAX_ITER+1):

        print("======== EPISODE {} BEGINS ========\n".format(cur_episode))
        logger.info("======== EPISODE {} BEGINS ========\n".format(cur_episode))

        # Creating output directory for the episode
        episode_dir = os.path.join(cfg.EXP_DIR, f'episode_{cur_episode}')
        if not os.path.exists(episode_dir):
            os.mkdir(episode_dir)
        cfg.EPISODE_DIR = episode_dir

        checkpoint_file, episode_train_time, epoch_times, best_train_acc, best_val_acc, test_acc = train_and_eval_model(cfg, cur_episode, lSet_loader, model, optimizer, test_loader,
                                               valSet_loader, plotting_manager)
        episode_training_times.append(episode_train_time)
        all_epoch_times.extend(epoch_times)
        
        # Store accuracies from current episode (will keep the last one)
        final_train_acc = best_train_acc
        final_val_acc = best_val_acc
        final_test_acc = test_acc

        # No need to perform active sampling in the last episode iteration
        if cur_episode == cfg.ACTIVE_LEARNING.MAX_ITER:
            # Save current lSet, uSet in the final episode directory
            data_obj.saveSet(lSet, 'lSet', cfg.EPISODE_DIR)
            data_obj.saveSet(uSet, 'uSet', cfg.EPISODE_DIR)
            break

        # Active Sample (algorithms handle their own state updates internally)
        al_start_time = time.time()
        lSet, lSet_loader, uSet, valSet_loader = active_sampling_part(cfg, checkpoint_file, cur_episode, data_obj, lSet,
                                                                      lSet_loader, train_data, uSet, valSet,
                                                                      valSet_loader, al_obj, train_labels=train_labels)
        al_time = time.time() - al_start_time
        episode_al_times.append(al_time)
        print(f"AL_STATS ep={cur_episode} time={al_time:.2f}s lSet={len(lSet)} uSet={len(uSet)}")
        
        _, ep_y = plotting_manager.get_episode_data()
        print('Current accuracy values: ', ep_y)

        if not cfg.ACTIVE_LEARNING.FINE_TUNE:
            # start model from scratch
            print('Starting model from scratch - ignoring existing weights.')
            model = mh.get_model(cfg, model_init_state=model_init_state)
            # Construct the optimizer
            optimizer = optim.construct_optimizer(cfg, model, opt_init_state=opt_init_state)

        if checkpoint_file is not None:
            os.remove(checkpoint_file)

    # Display final summary with timing and training metrics
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print("Command: python {}".format(" ".join(sys.argv)))
    print("\n--- Training Metrics ---")
    print("Final Train Accuracy:        {:.2f}%".format(final_train_acc))
    print("Final Validation Accuracy:   {:.2f}%".format(final_val_acc))
    print("Final Test Accuracy:         {:.2f}%".format(final_test_acc))
    print("Final Labeled Set Size:      {}".format(len(lSet)))
    print("\n--- Timing Statistics ---")
    print("Total Episodes:              {}".format(cfg.ACTIVE_LEARNING.MAX_ITER + 1))
    print("Total Epochs (all):          {}".format(len(all_epoch_times)))
    if episode_training_times and episode_al_times:
        avg_episode_time = np.mean(episode_training_times) + np.mean(episode_al_times)
        print("Average Episode Time:        {:.2f}s ({:.2f} min)".format(avg_episode_time, avg_episode_time/60))
        print("  - Avg Train Time:          {:.2f}s ({:.2f} min)".format(np.mean(episode_training_times), np.mean(episode_training_times)/60))
        print("  - Avg AL Sampling Time:    {:.2f}s ({:.2f} min)".format(np.mean(episode_al_times), np.mean(episode_al_times)/60))
    total_time = sum(episode_training_times) + sum(episode_al_times)
    print("Total Experiment Time:       {:.2f}s ({:.2f} min)".format(total_time, total_time/60))
    print("="*60 + "\n")


def active_sampling_part(cfg, checkpoint_file, cur_episode, data_obj, lSet, lSet_loader, train_data, uSet, valSet,
                         valSet_loader, al_obj, train_labels=None):
    print("======== ACTIVE SAMPLING ========\n")
    logger.info("======== ACTIVE SAMPLING ========\n")
    clf_model = mh.get_model(cfg, checkpoint_file)
    # Pass checkpoint_file to let algorithm handle its own state updates internally
    activeSet, new_uSet = al_obj.sample_from_uSet(
        clf_model, lSet, uSet, train_data, checkpoint_file=checkpoint_file
    )
    # Save current lSet, new_uSet and activeSet in the episode directory
    data_obj.saveSets(lSet, uSet, activeSet, cfg.EPISODE_DIR)
    # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
    lSet = np.append(lSet, activeSet)
    uSet = new_uSet
    distillation_enabled = getattr(cfg, 'DISTILLATION_TRAINING', False)
    lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data,
                                                return_index=distillation_enabled)
    valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    if valSet_loader is None:
        valSet_loader = lSet_loader
    if distillation_enabled:
        attach_distillation_targets(cfg, al_obj, train_labels, lSet, uSet, lSet_loader)
    print(
        "Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(
            cur_episode, len(lSet), len(uSet), len(activeSet)))
    logger.info(
        "Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(
            cur_episode, len(lSet), len(uSet), len(activeSet)))
    print("================================\n\n")
    logger.info("================================\n\n")
    return lSet, lSet_loader, uSet, valSet_loader


def train_and_eval_model(cfg, cur_episode, lSet_loader, model, optimizer, test_loader, valSet_loader, plotting_manager):
    # Train model
    train_start_time = time.time()
    print("======== TRAINING ========")
    logger.info("======== TRAINING ========")
    best_val_acc, best_val_epoch, best_train_acc, checkpoint_file, model, epoch_times = train_model(lSet_loader, valSet_loader, model, optimizer, cfg, plotting_manager)
    train_time = time.time() - train_start_time
    print("Best Validation Accuracy: {}\nBest Epoch: {}\n".format(round(best_val_acc, 4), best_val_epoch))
    logger.info("EPISODE {} Best Validation Accuracy: {}\tBest Epoch: {}\n".format(cur_episode, round(best_val_acc, 4),
                                                                                   best_val_epoch))
    # Test best model checkpoint
    print("======== TESTING ========\n")
    logger.info("======== TESTING ========\n")
    test_acc = test_model(test_loader, checkpoint_file, cfg, cur_episode, model, plotting_manager)
    print("Test Accuracy: {}.\n".format(round(test_acc, 4)))
    logger.info("EPISODE {} Test Accuracy {}.\n".format(cur_episode, test_acc))
    
    return checkpoint_file, train_time, epoch_times, best_train_acc, best_val_acc, test_acc


def train_model(train_loader, val_loader, model, optimizer, cfg, plotting_manager):

    if cfg.MODEL.USE_1NN:  # TODO move to train_model function
        # Fit 1-NN on the full labeled set (not just the first batch)
        def _collect_full(loader):
            xs, ys = [], []
            for xb, yb in loader:
                # DataLoader may return numpy arrays or torch tensors
                if hasattr(xb, "detach"):
                    xb = xb.detach().cpu().numpy()
                if hasattr(yb, "detach"):
                    yb = yb.detach().cpu().numpy()
                xs.append(xb)
                ys.append(yb)
            return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

        train_data, train_labels = _collect_full(train_loader)
        print(f"1NN fit: train_data={train_data.shape} labels={train_labels.shape}")
        model.fit(train_data, train_labels)

        val_data, val_labels = _collect_full(val_loader)
        print(f"1NN eval: val_data={val_data.shape} labels={val_labels.shape}")
        preds = model.predict(val_data)
        test_acc = 100. * (preds == np.array(val_labels)).mean()

        # For 1NN, train_acc is assumed to be perfect
        train_acc = 100.0
        return test_acc, 0, train_acc, None, model, []

    start_epoch = 0
    loss_fun = losses.get_loss_fun()

    # Create meters
    train_meter = TrainMeter(len(train_loader))
    val_meter = ValMeter(len(val_loader))

    # Perform the training loop
    logger.info('Start epoch: {}'.format(start_epoch + 1))
    val_set_acc = 0.

    temp_best_val_acc = 0.
    temp_best_val_epoch = 0

    # Best checkpoint model and optimizer states
    best_model_state = None
    best_opt_state = None

    val_acc_epochs_x = []
    val_acc_epochs_y = []
    
    temp_best_train_acc = 0.
    epoch_times = []  # Track time for each epoch

    clf_train_iterations = cfg.OPTIM.MAX_EPOCH * int(len(train_loader)/cfg.TRAIN.BATCH_SIZE)
    clf_change_lr_iter = clf_train_iterations // 25
    clf_iter_count = 0

    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):

        # Train for one epoch
        train_loss, train_acc, clf_iter_count, epoch_time = train_epoch(train_loader, model, loss_fun, optimizer, train_meter, \
                                        cur_epoch, cfg, clf_iter_count, clf_change_lr_iter, clf_train_iterations, plotting_manager)
        epoch_times.append(epoch_time)

        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            nu.compute_precise_bn_stats(model, train_loader)


        # Model evaluation
        if is_eval_epoch(cur_epoch):
            # Original code[PYCLS] passes on testLoader but we want to compute on val Set
            val_loader.dataset.no_aug = True
            val_set_err = test_epoch(val_loader, model, val_meter, cur_epoch)
            val_set_acc = 100. - val_set_err
            val_loader.dataset.no_aug = False
            if temp_best_val_acc < val_set_acc:
                temp_best_val_acc = val_set_acc
                temp_best_val_epoch = cur_epoch + 1
                temp_best_train_acc = train_acc

                # Save best model and optimizer state for checkpointing
                model.eval()

                best_model_state = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
                best_opt_state = optimizer.state_dict()

                model.train()

            # Since we start from 0 epoch
            val_acc_epochs_x.append(cur_epoch+1)
            val_acc_epochs_y.append(val_set_acc)

        plotting_manager.add_epoch(cur_epoch+1, train_loss)
        print('\rTraining Epoch: {}/{}\tTrain Loss: {}\tTrain Acc: {:.2f}%\tVal Accuracy: {:.2f}%'.format(cur_epoch+1, cfg.OPTIM.MAX_EPOCH, round(train_loss, 4), train_acc, val_set_acc), end="")

    # Save the best model checkpoint (Episode level)
    checkpoint_file = cu.save_checkpoint(info="vlBest_acc_"+str(int(temp_best_val_acc)), \
        model_state=best_model_state, optimizer_state=best_opt_state, epoch=temp_best_val_epoch, cfg=cfg)

    print('\nWrote Best Model Checkpoint to: {}\n'.format(checkpoint_file.split('/')[-1]))
    logger.info('Wrote Best Model Checkpoint to: {}\n'.format(checkpoint_file))

    plotting_manager.reset_epoch_plots()

    best_val_acc = temp_best_val_acc
    best_val_epoch = temp_best_val_epoch
    best_train_acc = temp_best_train_acc

    return best_val_acc, best_val_epoch, best_train_acc, checkpoint_file, None, epoch_times


def test_model(test_loader, checkpoint_file, cfg, cur_episode, model, plotting_manager):

    test_meter = TestMeter(len(test_loader))

    if cfg.MODEL.USE_1NN:
        # # 1NN model does not have a predict method, so we use the fit method to train it
        data = test_loader.dataset.features
        labels = test_loader.dataset.targets
        preds = model.predict(data)
        test_acc = 100. * (preds == np.array(labels)).mean()
    else:
        model = mh.get_model(cfg, checkpoint_file)
        test_err = test_epoch(test_loader, model, test_meter, cur_episode)
        test_acc = 100. - test_err

    plotting_manager.add_episode(cur_episode, test_acc)
    print(f"OUTPUT FOLDER IS {cfg.EXP_DIR}")
    plotting_manager.plot_episode_results(dataset_name=cfg.DATASET.NAME, out_dir=cfg.EXP_DIR)
    plotting_manager.save_episode_data(out_dir=cfg.EXP_DIR)

    return test_acc


def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch, cfg, clf_iter_count, clf_change_lr_iter, clf_max_iter, plotting_manager):
    """Performs one epoch of training."""
    epoch_start_time = time.time()

    # Shuffle the data
    if cfg.NUM_GPUS>1:  train_loader.sampler.set_epoch(cur_epoch)

    # Update the learning rate
    # Currently we only support LR schedules for only 'SGD' optimizer
    lr = optim.get_epoch_lr(cfg, cur_epoch)
    if cfg.OPTIM.TYPE == "sgd":
        optim.set_lr(optimizer, lr)

    if torch.cuda.is_available():
        model.cuda()

    # Enable training mode
    model.train()
    train_meter.iter_tic() #This basically notes the start time in timer class defined in utils/timer.py

    distillation_enabled = (
        getattr(cfg, 'DISTILLATION_TRAINING', False)
        and getattr(train_loader.dataset, 'soft_targets', None) is not None
    )

    len_train_loader = len(train_loader)
    for cur_iter, batch in enumerate(train_loader):
        if distillation_enabled:
            inputs, labels, idx = batch
        else:
            inputs, labels = batch
        #ensuring that inputs are floatTensor as model weights are
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        preds = model(inputs)
        # Compute the loss
        if distillation_enabled:
            dataset = train_loader.dataset
            idx_device = dataset.soft_targets.device
            idx = idx.to(idx_device)

            soft_targets_batch = dataset.soft_targets[idx].to(dtype=preds.dtype)
            distill_mask_batch = dataset.distill_mask[idx]
            distill_weights_batch = dataset.distill_weights[idx]
            labeled_mask_batch = (
                dataset.labeled_mask.to(idx_device)[idx]
                if dataset.labeled_mask is not None
                else torch.ones_like(distill_mask_batch)
            )

            # Only unlabeled, gated points contribute to the distillation loss --
            # labeled points already have ground-truth in the CE term below.
            distill_only_mask = distill_mask_batch & ~labeled_mask_batch

            ce_loss = loss_fun(preds, labels)

            distill_temp = getattr(cfg, 'DISTILLATION_TEMPERATURE', 3.0)
            distill_loss_raw = losses.distillation_loss(
                preds, soft_targets_batch, distill_only_mask, temperature=distill_temp)

            distill_factor = getattr(cfg, 'DISTILL_FACTOR', 1.0)
            if distill_only_mask.any():
                avg_weight = distill_weights_batch[distill_only_mask].mean()
                loss = ce_loss + distill_loss_raw * distill_factor * avg_weight
            else:
                loss = ce_loss
        else:
            loss = loss_fun(preds, labels)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the parametersSWA
        optimizer.step()
        # Compute the errors
        top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err = loss.item(), top1_err.item()
        if cur_iter != 0 and cur_iter%19 == 0:
            plotting_manager.add_iteration((cur_epoch)*len_train_loader + cur_iter, loss)
            print('Training Epoch: {}/{}\tIter: {}/{}'.format(cur_epoch+1, cfg.OPTIM.MAX_EPOCH, cur_iter, len(train_loader)), end='\r')

        train_meter.iter_toc()
        train_meter.update_stats(top1_err=top1_err, loss=loss, \
            lr=lr, mb_size=inputs.size(0) * cfg.NUM_GPUS)
        # Don't log iter stats to avoid interfering with progress line
        # train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    
    # Calculate train accuracy for the epoch
    # Note: topk_errors returns percentages (0-100), so num_top1_mis accumulates percentage*mb_size
    # Therefore num_top1_mis/num_samples gives the average error percentage
    train_acc = 100.0 - (train_meter.num_top1_mis / train_meter.num_samples)
    
    epoch_time = time.time() - epoch_start_time
    
    train_meter.reset()
    return loss, train_acc, clf_iter_count, epoch_time


@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""
    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    test_meter.iter_tic()

    misclassifications = 0.
    totalSamples = 0.

    for cur_iter, (inputs, labels) in enumerate(test_loader):
        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            inputs = inputs.type(torch.cuda.FloatTensor)
            preds = model(inputs)
            top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
            top1_err = top1_err.item()
            # Multiply by Number of GPU's as top1_err is scaled by 1/Num_GPUs
            misclassifications += top1_err * inputs.size(0) * cfg.NUM_GPUS
            totalSamples += inputs.size(0)*cfg.NUM_GPUS
            test_meter.iter_toc()
            # Update and log stats
            test_meter.update_stats(
                top1_err=top1_err, mb_size=inputs.size(0) * cfg.NUM_GPUS
            )
            test_meter.log_iter_stats(cur_epoch, cur_iter)
            test_meter.iter_tic()
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()

    return misclassifications/totalSamples




if __name__ == "__main__":
    # Parse command line arguments and setup config in one call
    args = argparser().parse_args()
    cfg_instance = setup_cfg(args)
    # Run main with the initialized config
    main(cfg_instance)
