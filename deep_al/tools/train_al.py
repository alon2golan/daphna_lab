import os
import sys
from datetime import datetime
from pathlib import Path
import argparse
import numpy as np
import time

# Ensure repo root is in sys.path for clean imports
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
from copy import deepcopy

from deep_al.pycls.al.active_learning import ActiveLearning
from deep_al.pycls.noise.estimating_delta_nu import estimate_nu
from deep_al.pycls.noise.nu_prime_cache import lookup_nu_for_tau, lookup_tau_for_nu
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
from deep_al.pycls.noise.annotator import Annotator
from deep_al.pycls.noise.annotator_manager import AnnotatorManager
from deep_al.pycls.noise.noisy_label_wrapper import NoisyLabelDatasetWrapper
from reporting.result_writer import ResultWriter

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
    parser.add_argument('--matrix_dtype', type=str, default=None, choices=['float32', 'float16'], help="Choose torch dtype: float32 or float16")
    parser.add_argument('--max_epoch', default=None, type=int, help='Max training epochs per AL round')
    # Equilibrium noise rate (nu') / single noisy annotator strategy
    parser.add_argument('--tau', type=float, default=None,
                        help='Cost ratio tau = cost_noisy / cost_oracle. When provided, nu\' is computed via '
                             'estimate_nu (or taken from --nu_prime) and cfg is reconfigured to a single noisy '
                             'annotator with noise_rate=nu\' and cost=tau.')
    parser.add_argument('--nu_prime', type=float, default=None,
                        help="nu' (equilibrium noise rate). Used together with --tau: pass a float to use directly, "
                             "or omit to auto-compute via estimate_nu.")
    parser.add_argument('--target_nu', type=float, default=None,
                        help='Target noise level (0-1). Looks up the breakeven tau from the nu_prime cache '
                             'so you can ask: at what cost ratio does a noisy annotator with this noise level '
                             'become worth using? Mutually exclusive with --tau.')
    parser.add_argument('--sweep-name', default=None, type=str,
                        help='Human-readable sweep name (auto-generated by sweep_train_al.py)')
    return parser


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
        train_spare_inds = np.sort(np.load(os.path.join(_repo_root, f"results/t-sne/{cfg['DATASET']['NAME']}_train_sparse_indices.npy")))
        test_spare_inds = np.sort(np.load(os.path.join(_repo_root, f"results/t-sne/{cfg['DATASET']['NAME']}_test_sparse_indices.npy")))

        train_data.features = train_data.features[train_spare_inds.astype(int)]
        train_data.targets = np.array(train_data.targets)[train_spare_inds].tolist()
        train_data.data = train_data.data[train_spare_inds.astype(int)]

        test_data.features = test_data.features[test_spare_inds]
        test_data.targets = np.array(test_data.targets)[test_spare_inds].tolist()
        test_data.data = test_data.data[test_spare_inds.astype(int)]

    return data_obj, train_data, test_data, train_size, test_size

def get_annotators(cfg, args, train_data):
    """Initialize annotators based on configuration.

    Always returns a non-empty list.  When annotators are disabled (or in oracle
    mode), a single perfect oracle annotator is returned so that the rest of the
    pipeline can remain annotator-manager-aware without any None checks.

    Args:
        cfg: Configuration object
        args: Command line arguments
        train_data: Training dataset

    Returns:
        List of Annotator objects (always at least one oracle annotator)
    """
    annotators = []
    
    if cfg.ANNOTATORS.ENABLED and cfg.ANNOTATORS.MODE != 'oracle':
        # Noisy annotators mode
        assert len(cfg.ANNOTATORS.NOISE_RATES) == cfg.ANNOTATORS.NUM_ANNOTATORS, \
            f"Number of noise rates ({len(cfg.ANNOTATORS.NOISE_RATES)}) must match NUM_ANNOTATORS ({cfg.ANNOTATORS.NUM_ANNOTATORS})"
        assert len(cfg.ANNOTATORS.COSTS) == cfg.ANNOTATORS.NUM_ANNOTATORS, \
            f"Number of costs ({len(cfg.ANNOTATORS.COSTS)}) must match NUM_ANNOTATORS ({cfg.ANNOTATORS.NUM_ANNOTATORS})"

        for i, (noise_rate, cost) in enumerate(zip(cfg.ANNOTATORS.NOISE_RATES, cfg.ANNOTATORS.COSTS)):
            annotator = Annotator(
                dataset=train_data,
                noise_type=cfg.ANNOTATORS.NOISE_TYPE,
                noise_rate=noise_rate,
                cost=cost,
                seed=args.seed + i,
                save_dir=cfg.EXP_DIR
            )
            annotators.append(annotator)
            print(f"Initialized Annotator {i}: noise_rate={noise_rate}, cost={cost}, accuracy={annotator.accuracy}")
    else:
        # Oracle / disabled: single perfect annotator (gives GT labels, zero cost)
        annotator = Annotator(
            dataset=train_data,
            noise_type="clean",
            noise_rate=0.0,
            cost=0.0,
            seed=args.seed,
            save_dir=cfg.EXP_DIR
        )
        annotators.append(annotator)
        if cfg.ANNOTATORS.ENABLED:
            print("Initialized Oracle Annotator (clean labels, cost=0)")
        else:
            print("Annotators disabled - falling back to oracle annotator (true labels)")

    return annotators

def apply_nu_prime_strategy(cfg, args, annotators, annotator_manager, train_data=None):
    """Apply equilibrium noise rate (nu') strategy for 2-annotator setup.
    
    This function is designed for the special case where there are exactly 2 annotators:
    - One oracle annotator (noise_rate = 0.0)
    - One noisy annotator (noise_rate > 0.0)
    
    nu' (equilibrium noise rate) can be supplied in two ways:
    - Directly as a float via args.nu_prime (>= 0)
    - Auto-computed via estimate_nu when args.nu_prime is None or negative;
      tau is derived automatically as cost_noisy / cost_oracle from cfg.ANNOTATORS.COSTS
    
    Based on the comparison with the noisy annotator's noise rate, it configures the
    AnnotatorManager to always select the appropriate annotator:
    - If noisy_noise_rate > nu': Always use oracle
    - If noisy_noise_rate <= nu': Always use noisy annotator
    
    Args:
        cfg: Configuration object
        args: Command line arguments
        annotators: List of annotator objects
        annotator_manager: AnnotatorManager instance
        train_data: Training dataset (required when computing nu' via estimate_nu)
        
    Returns:
        Updated annotator_manager with appropriate selection strategy
    """
    nu_prime_provided = args.nu_prime is not None and args.nu_prime >= 0

    # Only apply if annotators are enabled
    if not (cfg.ANNOTATORS.ENABLED and annotators is not None):
        return annotator_manager
    
    print("\n" + "=" * 60)
    print("  Equilibrium Noise Rate (nu') Strategy")
    print("=" * 60)
    
    try:
        # Verify we have exactly 2 annotators
        if len(annotators) != 2:
            print(f"\nWarning: nu' strategy requires exactly 2 annotators (oracle + noisy)")
            print(f"    Found {len(annotators)} annotators. Skipping nu' strategy.\n")
            print("=" * 60 + "\n")
            return annotator_manager
        
        # Identify oracle and noisy annotators
        oracle_idx = None
        noisy_idx = None
        noisy_rate = None
        
        for idx, ann in enumerate(annotators):
            if ann.noise_rate == 0.0:
                oracle_idx = idx
            else:
                noisy_idx = idx
                noisy_rate = ann.noise_rate
        
        # Verify we have one of each
        if oracle_idx is None or noisy_idx is None:
            print(f"\nWarning: nu' strategy requires 1 oracle (noise=0) and 1 noisy annotator")
            print(f"    Setup not compatible. Skipping nu' strategy.\n")
            print("=" * 60 + "\n")
            return annotator_manager
        
        # Resolve nu' - either from CLI float or by running estimate_nu
        if nu_prime_provided:
            nu_prime = args.nu_prime
            print(f"\nUsing provided nu' = {nu_prime:.4f} (from --nu_prime argument)")
        else:
            if train_data is None:
                print(f"\nWarning: nu' auto-computation requested but train_data not provided. Skipping nu' strategy.\n")
                print("=" * 60 + "\n")
                return annotator_manager
            # Derive tau from cost ratio: tau = cost_noisy / cost_oracle
            tau = annotators[noisy_idx].cost / annotators[oracle_idx].cost
            print(f"\nComputing nu' via estimate_nu (tau={tau:.4f} from cost ratio, budget={args.budget}, "
                  f"num_classes={cfg.MODEL.NUM_CLASSES}, dataset={cfg.DATASET.NAME})...")
            _, nu_prime, _ = estimate_nu(
                dataset=deepcopy(train_data),
                tau=tau,
                budget=args.budget,
                num_classes=cfg.MODEL.NUM_CLASSES,
                dataset_name=cfg.DATASET.NAME,
                seed=args.seed,
                cluster_labels=True,
                noise_mode="random",
                epochs=100,
                batch_size=32,
                epsilon=0.01,
                test_size=0.2,
                al_method=cfg.ACTIVE_LEARNING.SAMPLING_FN,
            )
            print(f"Computed nu' = {nu_prime:.4f}")

        print(f"\nAnnotator Setup:")
        print(f"  - Annotator {oracle_idx}: Oracle (noise_rate=0.0, cost={annotators[oracle_idx].cost})")
        print(f"  - Annotator {noisy_idx}: Noisy (noise_rate={noisy_rate:.4f}, cost={annotators[noisy_idx].cost})")
        print(f"\nEquilibrium Threshold:")
        print(f"  - nu' = {nu_prime:.4f}")

        # Decision: compare noisy annotator's noise rate vs nu'
        if noisy_rate > nu_prime:
            print(f"\nNoisy annotator noise ({noisy_rate:.4f}) > nu' ({nu_prime:.4f})")
            print("    Noise level exceeds equilibrium threshold!")
            print(f"    Selection Strategy: Always use Oracle (Annotator {oracle_idx})\n")
            
            annotator_manager.force_annotator(oracle_idx)
            
            logger.info(f"nu' strategy: Using oracle (noise={noisy_rate} > nu_prime={nu_prime})")
        else:
            print(f"\nNoisy annotator noise ({noisy_rate:.4f}) <= nu' ({nu_prime:.4f})")
            print("    Noise level is within acceptable range")
            print(f"    Selection Strategy: Always use Noisy Annotator (Annotator {noisy_idx})\n")
            
            annotator_manager.force_annotator(noisy_idx)
            
            logger.info(f"nu' strategy: Using noisy annotator (noise={noisy_rate} <= nu_prime={nu_prime})")
    
    except Exception as e:
        print(f"\nWarning: Failed to apply nu' strategy: {e}")
        print("    Continuing with original annotator configuration\n")
        logger.warning(f"nu' strategy failed: {e}")
    
    print("=" * 60 + "\n")
    return annotator_manager


def configure_nu_prime_annotator(cfg, args, train_data=None):
    """Reconfigure cfg to a single noisy annotator with noise_rate=nu' and cost=tau.

    Triggered when --tau or --target_nu is passed on the CLI.

    Resolution order for (tau, nu_prime):
      --tau + --nu_prime  : both taken directly from CLI
      --tau only          : nu_prime looked up from cache; falls back to estimate_nu
      --target_nu only    : tau looked up from cache; nu_prime = target_nu
      Neither             : error

    The cfg is mutated in-place:
        ANNOTATORS.ENABLED         <- True
        ANNOTATORS.NUM_ANNOTATORS  <- 1
        ANNOTATORS.NOISE_RATES     <- [nu_prime]
        ANNOTATORS.COSTS           <- [tau]
        ANNOTATORS.MODE            <- 'single'

    Args:
        cfg: Configuration object (mutated in-place)
        args: Command line arguments
        train_data: Training dataset (required only when falling back to estimate_nu)
    """
    print("\n" + "=" * 60)
    print("  Single Noisy Annotator via Nash Equilibrium nu'")
    print("=" * 60)

    dataset_name = cfg.DATASET.NAME.lower()
    al_method    = cfg.ACTIVE_LEARNING.SAMPLING_FN.lower()
    budget       = args.budget

    # ── Validate mutual exclusion ──────────────────────────────────────────────
    if args.tau is not None and getattr(args, 'target_nu', None) is not None:
        raise ValueError("--tau and --target_nu are mutually exclusive. Use one or the other.")

    # ── Case 1: --target_nu supplied → look up tau from cache ─────────────────
    if getattr(args, 'target_nu', None) is not None:
        target_nu = args.target_nu
        print(f"\ntarget_nu = {target_nu:.4f} (from --target_nu argument)")
        tau = lookup_tau_for_nu(dataset_name, al_method, budget, target_nu)
        if tau is None:
            raise ValueError(
                f"No cache entry for (dataset={dataset_name}, al_method={al_method}, "
                f"budget={budget}). Run batch_utils/sweep_nu_prime.py first."
            )
        nu_prime = target_nu
        print(f"tau = {tau:.4f}  (from cache, interpolated for target_nu={target_nu:.4f})")
        print(f"nu' = {nu_prime:.4f}  (= target_nu)")

    # ── Case 2: --tau supplied ─────────────────────────────────────────────────
    else:
        tau = args.tau
        print(f"\ntau = {tau:.4f} (from --tau argument)")

        if args.nu_prime is not None and args.nu_prime >= 0:
            # Explicitly provided
            nu_prime = args.nu_prime
            print(f"nu' = {nu_prime:.4f} (from --nu_prime argument)")
        else:
            # Try cache first (fast)
            nu_prime = lookup_nu_for_tau(dataset_name, al_method, budget, tau)
            if nu_prime is not None:
                print(f"nu' = {nu_prime:.4f} (from cache, interpolated)")
            else:
                # Fall back to live estimate_nu (slow)
                if train_data is None:
                    raise ValueError(
                        "No cache entry and --nu_prime not provided; "
                        "train_data is required to auto-compute nu' via estimate_nu."
                    )
                print(f"Computing nu' via estimate_nu (tau={tau:.4f}, budget={budget}, "
                      f"num_classes={cfg.MODEL.NUM_CLASSES}, dataset={dataset_name})...")
                _, nu_prime, _ = estimate_nu(
                    dataset=deepcopy(train_data),
                    tau=tau,
                    budget=budget,
                    num_classes=cfg.MODEL.NUM_CLASSES,
                    dataset_name=dataset_name,
                    seed=args.seed,
                    cluster_labels=True,
                    noise_mode="random",
                    epochs=100,
                    batch_size=32,
                    epsilon=0.01,
                    test_size=0.2,
                    al_method=al_method,
                )
                print(f"nu' = {nu_prime:.4f} (computed via estimate_nu)")

    # Mutate cfg to a single annotator
    cfg.ANNOTATORS.ENABLED = True
    cfg.ANNOTATORS.NUM_ANNOTATORS = 1
    cfg.ANNOTATORS.NOISE_RATES = [nu_prime]
    cfg.ANNOTATORS.COSTS = [tau]
    cfg.ANNOTATORS.MODE = 'single'

    print(f"\nReconfigured cfg to single noisy annotator:")
    print(f"  noise_rate = {nu_prime:.4f}  (= nu')")
    print(f"  cost       = {tau:.4f}  (= tau)")
    print("=" * 60 + "\n")


def setup_experiment_directory(cfg):
    """Setup experiment directory structure and configuration.
    
    Args:
        cfg: Configuration object
        
    Returns:
        exp_dir: Path to the experiment directory
    """
    # Get the output directory ready (paths are already normalized from config.py)
    if not os.path.exists(cfg.OUT_DIR):
        os.mkdir(cfg.OUT_DIR)
    
    # Create "DATASET/MODEL TYPE" specific directory
    dataset_out_dir = os.path.join(cfg.OUT_DIR, cfg.DATASET.NAME, cfg.MODEL.TYPE)
    if not os.path.exists(dataset_out_dir):
        os.makedirs(dataset_out_dir)
    
    # Create day-specific directory
    now = datetime.now()
    day_dir = f'{now.year}_{now.month}_{now.day}'
    full_day_dir = os.path.join(dataset_out_dir, day_dir)
    if not os.path.exists(full_day_dir):
        os.mkdir(full_day_dir)
    
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


def main(cfg, args):
    plotting_manager = PlottingManager()
    print("RNG_SEED is set to {}".format(cfg.RNG_SEED))
    # Setting up GPU args
    use_cuda = (cfg.NUM_GPUS > 0) and torch.cuda.is_available()

    # Auto assign a RNG_SEED when not supplied a value
    if cfg.RNG_SEED is None:
        cfg.RNG_SEED = np.random.randint(100)

    # Setup experiment directory structure
    setup_experiment_directory(cfg)

    # Start result tracking (writes result.json; updated after each episode)
    writer = ResultWriter(cfg.EXP_DIR, args, cfg)

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
    
    # Initialize annotators and annotator manager (always — oracle by default when disabled)
    # When --tau or --target_nu is provided, compute nu' and rewrite cfg to a single noisy
    # annotator (noise_rate=nu', cost=tau) before get_annotators() is called.
    _use_nu_prime_cfg = args.tau is not None or getattr(args, 'target_nu', None) is not None
    if _use_nu_prime_cfg:
        configure_nu_prime_annotator(cfg, args, train_data)

    annotators = get_annotators(cfg, args, train_data)
    annotator_manager = AnnotatorManager(
        annotators=annotators,
        selection_strategy=cfg.ANNOTATORS.SELECTION_STRATEGY,
        seed=args.seed,
        save_dir=cfg.EXP_DIR
    )

    # Apply the 2-annotator nu' threshold strategy only when --tau/--target_nu are not used.
    if not _use_nu_prime_cfg:
        annotator_manager = apply_nu_prime_strategy(cfg, args, annotators, annotator_manager, train_data)

    # Always wrap train_data — NoisyLabelDatasetWrapper falls through to true labels
    # for oracle-annotated samples, so behaviour is identical when using a clean annotator.
    train_data_wrapped = NoisyLabelDatasetWrapper(train_data, annotator_manager=annotator_manager)

    if cfg.HIGH_BUDGET:
        seed = 42
        rng = np.random.default_rng(seed)
        lset_indices_from_uset = rng.choice(uSet.size, size=10000, replace=False)
        lset_mask = np.zeros_like(uSet, dtype=bool)
        lset_mask[lset_indices_from_uset] = True
        lSet = uSet[lset_mask]
        uSet = uSet[~lset_mask]

    al_obj = ActiveLearning(data_obj, cfg, lSet, plotting_manager, annotator_manager)

    if len(lSet) == 0:
        # No initial labeled set — run one AL round to bootstrap lSet
        episode_dir = os.path.join(cfg.EXP_DIR, 'episode_initial')
        os.makedirs(episode_dir, exist_ok=True)
        cfg.EPISODE_DIR = episode_dir
        lSet, lSet_loader, uSet, valSet_loader = active_sampling_part(
            cfg, None, 0, data_obj, lSet, None, train_data, train_data_wrapped,
            uSet, valSet, None, al_obj, annotator_manager
        )
    else:
        # Annotate the initial labeled set then build dataloaders
        initial_annotator_id = annotator_manager.select_annotator()
        result = annotator_manager.annotate_samples(lSet, annotator_id=initial_annotator_id)
        print(f"Initial labeled set ({len(lSet)} samples) annotated. Cost: {result['cost_spent']}")
        lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data_wrapped)
        valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)  # ValSet always uses true labels
        if valSet_loader is None:
            valSet_loader = lSet_loader

    print("Data Partitioning Complete. \nLabeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))
    logger.info("Labeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))

    test_loader = data_obj.getTestLoader(data=test_data, test_batch_size=cfg.TRAIN.BATCH_SIZE, seed_id=cfg.RNG_SEED)

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
        writer.record_episode(cur_episode, test_acc, best_val_acc, best_train_acc, len(lSet), episode_train_time)

        # No need to perform active sampling in the last episode iteration
        if cur_episode == cfg.ACTIVE_LEARNING.MAX_ITER:
            # Save current lSet, uSet in the final episode directory
            data_obj.saveSet(lSet, 'lSet', cfg.EPISODE_DIR)
            data_obj.saveSet(uSet, 'uSet', cfg.EPISODE_DIR)
            break

        # Active Sample (algorithms handle their own state updates internally)
        al_start_time = time.time()
        lSet, lSet_loader, uSet, valSet_loader = active_sampling_part(cfg, checkpoint_file, cur_episode, data_obj, lSet,
                                                                      lSet_loader, train_data, train_data_wrapped, uSet, valSet,
                                                                      valSet_loader, al_obj, annotator_manager)
        al_time = time.time() - al_start_time
        episode_al_times.append(al_time)
        
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

    writer.finish()

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

def active_sampling_part(cfg, checkpoint_file, cur_episode, data_obj, lSet, lSet_loader, train_data, train_data_wrapped, uSet, valSet,
                         valSet_loader, al_obj, annotator_manager):
    print("======== ACTIVE SAMPLING ========\n")
    logger.info("======== ACTIVE SAMPLING ========\n")
    clf_model = mh.get_model(cfg, checkpoint_file)

    num_samples, batch_annotator_id = annotator_manager.plan_annotation_round(cfg.ACTIVE_LEARNING.BUDGET_SIZE)
    print(f"Budget={cfg.ACTIVE_LEARNING.BUDGET_SIZE}, num_samples to query={num_samples}")

    # Algorithms that need labels during sampling (e.g. CoverageMatrixMethod) annotate
    # each point in-loop via annotator_manager; others ignore annotator_id entirely.
    activeSet, new_uSet = al_obj.sample_from_uSet(
        clf_model, lSet, uSet, train_data,
        checkpoint_file=checkpoint_file,
        num_samples=num_samples,
        annotator_id=batch_annotator_id
    )

    print(f"Active Learning selected {len(activeSet)} candidate samples")
    logger.info(f"Active Learning selected {len(activeSet)} candidate samples")

    # Annotate selected samples. If already annotated in-loop (e.g. CoverageMatrixMethod),
    # skip to avoid double-counting cost.
    already_annotated = all(int(s) in annotator_manager.annotated_labels for s in activeSet)
    if already_annotated:
        print(f"Annotation Complete (done in-loop): {len(activeSet)} samples, cumulative cost: {annotator_manager.cumulative_cost}")
        logger.info(f"Annotation done in-loop: {len(activeSet)} samples, cumulative cost: {annotator_manager.cumulative_cost}")
    else:
        annotation_result = annotator_manager.annotate_samples(activeSet, annotator_id=batch_annotator_id)
        activeSet = np.array(annotation_result['annotated_samples'])
        print(f"Annotation Complete: {len(activeSet)} samples, cost: {annotation_result['cost_spent']}, cumulative: {annotator_manager.cumulative_cost}")
        logger.info(f"Annotation: {len(activeSet)} samples, cost: {annotation_result['cost_spent']}, cumulative: {annotator_manager.cumulative_cost}")

    # Save current lSet, new_uSet and activeSet in the episode directory
    data_obj.saveSets(lSet, uSet, activeSet, cfg.EPISODE_DIR)

    annotator_manager.save_state(cfg.EPISODE_DIR)
    stats = annotator_manager.get_statistics()
    print(f"Annotator Statistics: total={stats['total_annotated']}, cost={stats['total_cost']}, usage={stats['annotator_counts']}")
    logger.info(f"Annotator Stats: {stats}")
    
    # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
    lSet = np.append(lSet, activeSet)
    uSet = new_uSet
    lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data_wrapped)
    valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)  # ValSet always uses true labels
    if valSet_loader is None:
        valSet_loader = lSet_loader
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

    len_train_loader = len(train_loader)
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        #ensuring that inputs are floatTensor as model weights are
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        preds = model(inputs)
        # Compute the loss
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
    main(cfg_instance, args)
