"""
Noise Module: Components for handling noisy annotations in active learning.

This module contains:
- Annotator: Models noisy annotators with configurable noise rates
- AnnotatorManager: Manages multiple annotators and selection strategies
- NoisyLabelDatasetWrapper: Wraps datasets to use noisy labels
- estimating_delta_nu: Utilities for equilibrium noise rate (ν') estimation
"""

from .annotator import Annotator
from .annotator_manager import AnnotatorManager
from .noisy_label_wrapper import NoisyLabelDatasetWrapper
from .estimating_delta_nu import estimate_nu

__all__ = [
    'Annotator',
    'AnnotatorManager',
    'NoisyLabelDatasetWrapper',
    'estimate_nu',
]
