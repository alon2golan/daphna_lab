# This file is modified from code implementation by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564

# ----------------------------------------------------------
# This file is modified from official pycls repository to adapt in AL settings.

"""Configuration file (powered by YACS)."""

import os

from yacs.config import CfgNode as CN

# Global config object
_C = CN()

# Example usage:
#   from core.config import cfg
cfg = _C

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1
# Output directory (will be created at the projec root)
_C.OUT_DIR = 'output'
# Experiment directory
_C.EXP_DIR = ''
# Episode directory
_C.EPISODE_DIR = ''
# Config destination (in OUT_DIR)
_C.CFG_DEST = 'config.yaml'
# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.RNG_SEED = None
# Folder name where best model logs etc are saved. "auto" creates a timestamp based folder 
_C.EXP_NAME = 'auto'
# Eval model type used for downstream evaluation (options: 'NN1', 'from_features', 'from_images')
_C.EVAL_MODEL_TYPE = 'from_features'
# Which GPU to run on
_C.GPU_ID = 0
# Log destination ('stdout' or 'file')
_C.LOG_DEST = 'file'
# Log period in iters
_C.LOG_PERIOD = 10

# ------------------------------------------------------------------------------#
# VAAL Options (Taken from https://arxiv.org/abs/1904.00370)
# ------------------------------------------------------------------------------#
_C.VAAL = CN()
_C.VAAL.TRAIN_VAAL = False
_C.VAAL.Z_DIM = 32
_C.VAAL.VAE_BS = 64
_C.VAAL.VAE_EPOCHS = 100
_C.VAAL.VAE_LR = 5e-4
_C.VAAL.DISC_LR = 5e-4
_C.VAAL.BETA = 1.0
_C.VAAL.ADVERSARY_PARAM = 1.0
_C.VAAL.IM_SIZE = 32

# ------------------------------------------------------------------------------#
# Ensemble Options
# ------------------------------------------------------------------------------#
_C.ENSEMBLE = CN()
_C.ENSEMBLE.NUM_MODELS = 3
_C.ENSEMBLE.SAME_MODEL = True
_C.ENSEMBLE.MODEL_TYPE = ['resnet18']

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
# Model type. 
# Choose from vgg style ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',]
# or from resnet style ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 
# 'wide_resnet50_2', 'wide_resnet101_2']
_C.MODEL.TYPE = ''
# Number of classes
_C.MODEL.NUM_CLASSES = 10
# Loss function (see pycls/models/loss.py for options)
_C.MODEL.LOSS_FUN = 'cross_entropy'
# Robust/noisy-label loss options
_C.MODEL.GCE_Q = 0.7
_C.MODEL.BOOTSTRAP_BETA = 0.8

# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CN()
# BN epsilon
_C.BN.EPS = 1e-5
# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1
# Precise BN stats
_C.BN.USE_PRECISE_STATS = False
_C.BN.NUM_SAMPLES_PRECISE = 1024
# Initialize the gamma of the final BN of each block to zero
_C.BN.ZERO_INIT_FINAL_GAMMA = False

# Use a different weight decay for BN layers
_C.BN.USE_CUSTOM_WEIGHT_DECAY = False
_C.BN.CUSTOM_WEIGHT_DECAY = 0.0

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.OPTIM = CN()
_C.OPTIM.TYPE = 'sgd'
# Learning rate ranges from BASE_LR to MIN_LR*BASE_LR according to the LR_POLICY
_C.OPTIM.BASE_LR = 0.1
_C.OPTIM.MIN_LR = 0.0
# Learning rate policy select from {'cos', 'exp', 'steps'}
_C.OPTIM.LR_POLICY = 'cos'
# Exponential decay factor
_C.OPTIM.GAMMA = 0.1
# Steps for 'steps' policy (in epochs)
_C.OPTIM.STEPS = []
# Learning rate multiplier for 'steps' policy
_C.OPTIM.LR_MULT = 0.1
# Maximal number of epochs
_C.OPTIM.MAX_EPOCH = 200
# Momentum
_C.OPTIM.MOMENTUM = 0.9
# Momentum dampening
_C.OPTIM.DAMPENING = 0.0
# Nesterov momentum
_C.OPTIM.NESTEROV = False
# L2 regularization
_C.OPTIM.WEIGHT_DECAY = 5e-4
# Start the warm up from OPTIM.BASE_LR * OPTIM.WARMUP_FACTOR
_C.OPTIM.WARMUP_FACTOR = 0.1
# Gradually warm up the OPTIM.BASE_LR over this number of epochs
_C.OPTIM.WARMUP_EPOCHS = 0

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
# Dataset and split
_C.TRAIN.DATASET = ''
_C.TRAIN.SPLIT = 'train'
_C.TRAIN.IMBALANCED = False
# Total mini-batch size
_C.TRAIN.BATCH_SIZE = 128
# Image size
_C.TRAIN.IM_SIZE = 224
_C.TRAIN.IM_CHANNELS = 3
# Evaluate model on test data every eval period epochs
_C.TRAIN.EVAL_PERIOD = 1
# Save model checkpoint every checkpoint period epochs
_C.TRAIN.CHECKPOINT_PERIOD = 1
# Resume training from the latest checkpoint in the output directory
_C.TRAIN.AUTO_RESUME = False
# Weights to start training from
_C.TRAIN.WEIGHTS = ''
_C.TRAIN.TRANSFER_EXP = False

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Dataset and split
_C.TEST.DATASET = ''
_C.TEST.SPLIT = 'val'
# Total mini-batch size
_C.TEST.BATCH_SIZE = 200
# Image size
_C.TEST.IM_SIZE = 256
# Weights to use for testing
_C.TEST.WEIGHTS = ''
# Saved model to use for testing
_C.TEST.MODEL_PATH = ''

# ---------------------------------------------------------------------------- #
# Random Augmentation options
# ---------------------------------------------------------------------------- #
_C.RANDAUG = CN()
_C.RANDAUG.ACTIVATE = False
_C.RANDAUG.N = 1
_C.RANDAUG.M = 5

# #-------------------------------------------------------------------------------#
# #  ACTIVE LEARNING options
# #-------------------------------------------------------------------------------#
_C.ACTIVE_LEARNING = CN()
_C.ACTIVE_LEARNING.SAMPLING_FN = 'random'  # 'entropy', 'margin', 'vaal', 'coreset', 'ensemble_var_R'
_C.ACTIVE_LEARNING.ACTIVATE = False
_C.ACTIVE_LEARNING.LSET_PATH = ''
_C.ACTIVE_LEARNING.USET_PATH = ''
_C.ACTIVE_LEARNING.VALSET_PATH = ''
_C.ACTIVE_LEARNING.MODEL_LOAD_DIR = ''
_C.ACTIVE_LEARNING.MODEL_SAVE_DIR = ''
_C.ACTIVE_LEARNING.DATA_SPLIT = 0
_C.ACTIVE_LEARNING.BUDGET_SIZE = 5000  # 10% of initial lSet
_C.ACTIVE_LEARNING.N_BINS = 500  # Used by UC_uniform
_C.ACTIVE_LEARNING.DROPOUT_ITERATIONS = 25  # Used by DBAL and BALD
_C.ACTIVE_LEARNING.INIT_L_RATIO = 0.1  # Initial labeled pool ration
_C.ACTIVE_LEARNING.MAX_ITER = 5  # Max AL iterations
_C.ACTIVE_LEARNING.MAX_DELTA = 1.1  # Max Delta for DCoM algorithm
_C.ACTIVE_LEARNING.DELTA_RESOLUTION = 0.05  # Delta resolution for the binary search in DCoM algorithm
_C.ACTIVE_LEARNING.FINE_TUNE = True  # continue after AL from existing model or from scratch

# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CN()
# Number of data loader workers per training process
_C.DATA_LOADER.NUM_WORKERS = 4
# Load data to pinned host memory
_C.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# CUDNN options
# ---------------------------------------------------------------------------- #
_C.CUDNN = CN()
# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN.BENCHMARK = False

# ---------------------------------------------------------------------------- #
# Dataset options
# ---------------------------------------------------------------------------- #
_C.DATASET = CN()
_C.DATASET.NAME = None
# Optional tag to specify which feature representation to use (e.g., 'alon', 'itai')
_C.DATASET.FEATURES_TAG = None
# For Tiny ImageNet dataset, ROOT_DIR must be set to the dataset folder ("data/tiny-imagenet-200/"). For others, the outder "data" folder where all datasets can be stored is expected.
_C.DATASET.ROOT_DIR = None
# Specifies the proportion of data in train set that should be considered as the validation data
_C.DATASET.VAL_RATIO = 0.1
# Data augmentation methods - 'simclr', 'randaug', 'hflip'
_C.DATASET.AUG_METHOD = 'hflip'
# Accepted Datasets
_C.DATASET.ACCEPTED = ['MNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'TINYIMAGENET', 'IMBALANCED_CIFAR10',
                       'IMBALANCED_CIFAR100', 'IMAGENET50', 'IMAGENET100', 'IMAGENET200', 'SCENARIO_A', 'HALF_MOON']


def assert_cfg():
    """Checks config values invariants."""
    assert not _C.OPTIM.STEPS or _C.OPTIM.STEPS[0] == 0, \
        'The first lr step must start at 0'
    assert _C.TRAIN.SPLIT in ['train', 'val', 'test'], \
        'Train split \'{}\' not supported'.format(_C.TRAIN.SPLIT)
    assert _C.TRAIN.BATCH_SIZE % _C.NUM_GPUS == 0, \
        'Train mini-batch size should be a multiple of NUM_GPUS.'
    assert _C.TEST.SPLIT in ['train', 'val', 'test'], \
        'Test split \'{}\' not supported'.format(_C.TEST.SPLIT)
    assert _C.TEST.BATCH_SIZE % _C.NUM_GPUS == 0, \
        'Test mini-batch size should be a multiple of NUM_GPUS.'

    # our assertions
    if _C.ACTIVE_LEARNING.SAMPLING_FN == "uncertainty_uniform_discretize":
        assert _C.ACTIVE_LEARNING.N_BINS != 0, \
            "The number of bins used cannot be 0. Please provide a number >0 for {} sampling function" \
                .format(_C.ACTIVE_LEARNING.SAMPLING_FN)


def custom_dump_cfg(temp_cfg):
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(temp_cfg.EXP_DIR, temp_cfg.CFG_DEST)
    with open(cfg_file, 'w') as f:
        _C.dump(stream=f)


def dump_cfg(cfg):
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(cfg.EXP_DIR, cfg.CFG_DEST)
    with open(cfg_file, 'w') as f:
        cfg.dump(stream=f)


def load_cfg(out_dir, cfg_dest='config.yaml'):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def initialize_cfg(cfg_file, args_dict):
    """
    Unified config initialization function that merges all three config sources.

    Priority order (highest to lowest):
    1. argparse arguments (args_dict) - highest priority
    2. yaml config file (cfg_file)
    3. config.py defaults (_C) - lowest priority

    Args:
        cfg_file (str): Path to the YAML configuration file
        args_dict (dict): Dictionary of argparse arguments to override configs

    Returns:
        cfg: The merged configuration object
    """
    # Start with default config from config.py
    cfg = _C.clone()

    # Step 1: Merge YAML config file (lower priority than argparse)
    if cfg_file:
        cfg.merge_from_file(cfg_file)

    # Step 2: Apply argparse arguments (highest priority - these override everything)
    if args_dict:
        # Filter out None values to only apply explicitly set arguments
        for key, value in args_dict.items():
            if value is not None:
                # Handle nested config attributes (e.g., 'ACTIVE_LEARNING.SAMPLING_FN')
                if isinstance(key, str) and '.' in key:
                    parts = key.split('.')
                    config_obj = cfg
                    for part in parts[:-1]:
                        config_obj = getattr(config_obj, part)
                    setattr(config_obj, parts[-1], value)
                else:
                    # Direct attribute assignment
                    try:
                        setattr(cfg, key, value)
                    except Exception:
                        # If direct assignment fails, try nested access
                        pass

    return cfg


def get_config_aliases():
    """
    Returns a dictionary of configuration aliases.
    Each alias maps a single argument value to multiple config parameter updates.
    This allows setting multiple related config params with one argument.

    Example:
        aliases['eval_model_type']['from_features'] = {
            'MODEL.USE_1NN': False,
            'MODEL.LINEAR_FROM_FEATURES': True,
            ...
        }
    """
    aliases = {
        'eval_model_type': {
            'NN1': {
                'MODEL.USE_1NN': True,
                'MODEL.LINEAR_FROM_FEATURES': True,
                'MODEL.LINEAR_FROM_IMAGES': False,
            },
            'from_features': {
                'MODEL.USE_1NN': False,
                'MODEL.LINEAR_FROM_FEATURES': True,
                'MODEL.LINEAR_FROM_IMAGES': False,
            },
            'from_images': {
                'MODEL.USE_1NN': False,
                'MODEL.LINEAR_FROM_FEATURES': False,
                'MODEL.LINEAR_FROM_IMAGES': True,
            },
        },
        'debug': {
            True: {
                'OPTIM.MAX_EPOCH': 1,
                'RNG_SEED': 0,
            },
        }
    }
    return aliases


def apply_config_aliases(alias_name, alias_value, aliases):
    """
    Applies configuration aliases to get all related config parameter updates.

    Args:
        alias_name (str): The name of the alias (e.g., 'eval_model_type')
        alias_value (str): The value of the alias (e.g., 'from_features')
        aliases (dict): The aliases dictionary from get_config_aliases()

    Returns:
        dict: Dictionary of config parameters to update, or empty dict if alias not found
    """
    if alias_name not in aliases:
        return {}

    alias_dict = aliases[alias_name]
    if alias_value not in alias_dict:
        return {}

    return alias_dict[alias_value]


def merge_alias_updates(args_dict, aliases_to_apply, overwrite=False):
    """
    Merges alias updates into the arguments dictionary.

    Args:
        args_dict (dict): The arguments dictionary to merge updates into
        aliases_to_apply (list): List of tuples (alias_name, alias_value, aliases_dict)
        overwrite (bool): If True, aliases overwrite existing args_dict values.
                         If False, aliases only set values that don't exist in args_dict.
                         Default is False (non-invasive mode).

    Returns:
        dict: The updated args_dict
    """
    for alias_name, alias_value, aliases_dict in aliases_to_apply:
        updates = apply_config_aliases(alias_name, alias_value, aliases_dict)

        for key, value in updates.items():
            if overwrite or key not in args_dict:
                args_dict[key] = value

    return args_dict


def build_args_dict(args):
    """
    Builds the complete arguments dictionary from parsed command-line arguments,
    including alias updates.

    This function consolidates:
    1. Direct argument mapping to config keys
    2. Alias-based config parameter updates

    Args:
        args: Parsed arguments from argparser()

    Returns:
        dict: The complete args_dict with all direct arguments and alias updates
    """
    debug = args.debug

    # Build base arguments dictionary
    args_dict = {
        'EXP_NAME': args.exp_name,
        'ACTIVE_LEARNING.SAMPLING_FN': args.al,
        'ACTIVE_LEARNING.BUDGET_SIZE': args.budget,
        'ACTIVE_LEARNING.INITIAL_DELTA': args.initial_delta,
        'ACTIVE_LEARNING.INITIAL_SIGMA': args.initial_sigma,
        'KERNEL_TYPE': args.kernel_type,
        'SOFT_BORDER_VAL': args.soft_border_val,
        'DIFF_METHOD': args.diff_method,
        'CONT_METHOD': args.cont_method,
        'DECREASING_ALPHA': args.decrease_alpha,
        'DEBUG': debug,
        'HIGH_BUDGET': args.high_budget,
        'CONFIDENCE_METHOD': args.confidence_method,
        'RNG_SEED': args.seed,
        'ALPHA': args.alpha,
        'NORM_IMPORTANCE': args.norm_importance,
        'SPARSE_DS': args.sparse_ds,
        'ACTIVE_LEARNING.A_LOGISTIC': args.a_logistic,
        'ACTIVE_LEARNING.K_LOGISTIC': args.k_logistic,
        'EVAL_MODEL_TYPE': args.eval_model_type,
        'ACTIVE_LEARNING.MAX_ITER': args.max_iter,
        'SPARSE_K': args.sparse_K,
        'MATRIX_DTYPE': args.matrix_dtype,
        'DATASET.FEATURES_TAG': args.features_tag,
    }

    # Add max_epoch if provided
    if args.max_epoch is not None:
        args_dict['OPTIM.MAX_EPOCH'] = args.max_epoch

    # Get config aliases
    aliases = get_config_aliases()

    # Apply eval_model_type alias with non-invasive mode (doesn't overwrite existing values)
    eval_model_updates = [('eval_model_type', args.eval_model_type, aliases)]
    merge_alias_updates(args_dict, eval_model_updates, overwrite=False)

    # Apply debug alias with invasive mode (overwrites existing values)
    debug_updates = [('debug', debug, aliases)]
    merge_alias_updates(args_dict, debug_updates, overwrite=True)

    return args_dict


def normalize_cfg_paths(cfg):
    """
    Normalize configuration paths from relative to absolute paths from repo root.

    Converts relative paths to absolute paths using the repository root as the base.
    This ensures paths work correctly regardless of the current working directory.

    Normalizes:
    - DATASET.ROOT_DIR: Dataset root directory
    - OUT_DIR: Output directory for experiments

    Args:
        cfg: Configuration object to normalize

    Returns:
        cfg: The configuration object with normalized paths
    """
    # Get repo root (parent of deep_al/pycls/core/)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    # Normalize DATASET.ROOT_DIR if set and relative
    if cfg.DATASET.ROOT_DIR is not None:
        if not os.path.isabs(cfg.DATASET.ROOT_DIR):
            cfg.DATASET.ROOT_DIR = os.path.join(repo_root, cfg.DATASET.ROOT_DIR)

    # Normalize OUT_DIR (convert relative 'output' to absolute path)
    if cfg.OUT_DIR and not os.path.isabs(cfg.OUT_DIR):
        cfg.OUT_DIR = os.path.join(repo_root, cfg.OUT_DIR)

    return cfg


def setup_cfg(args):
    """
    One-shot configuration setup from command-line arguments.

    Takes parsed arguments, builds the complete args_dict (including aliases),
    initializes the config with proper priority ordering, and prints RNG_SEED.

    This is the main entry point for config setup in training scripts.

    Args:
        args: Parsed arguments from argparser() (must have cfg_file attribute)

    Returns:
        cfg: The initialized and merged configuration object
    """
    # Build args_dict from arguments (handles aliases, special cases)
    args_dict = build_args_dict(args)

    # Initialize config with unified function
    # Priority: argparse (highest) > cfg_file > config.py defaults (lowest)
    cfg_instance = initialize_cfg(args.cfg_file, args_dict)

    # Normalize paths from relative to absolute (based on repo root)
    normalize_cfg_paths(cfg_instance)

    return cfg_instance

