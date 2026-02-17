"""
Noise Module: Components for handling noisy annotations in active learning.

This module contains:
- Annotator: Models noisy annotators with configurable noise rates
- AnnotatorManager: Manages multiple annotators and selection strategies
- NoisyLabelDatasetWrapper: Wraps datasets to use noisy labels
- nu_prime_utils: Utilities for equilibrium noise rate (ν') estimation
"""

from .annotator import Annotator
from .annotator_manager import AnnotatorManager
from .noisy_label_wrapper import NoisyLabelDatasetWrapper
from .nu_prime_utils import (
    load_nu_prime,
    save_nu_prime,
    extract_features,
    get_pseudo_labels_kmeans,
    add_symmetric_noise,
    add_asymmetric_noise,
    simple_train_epoch,
    simple_eval_epoch
)

__all__ = [
    'Annotator',
    'AnnotatorManager',
    'NoisyLabelDatasetWrapper',
    'load_nu_prime',
    'save_nu_prime',
    'extract_features',
    'get_pseudo_labels_kmeans',
    'add_symmetric_noise',
    'add_asymmetric_noise',
    'simple_train_epoch',
    'simple_eval_epoch',
]
