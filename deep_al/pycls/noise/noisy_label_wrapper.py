"""
NoisyLabelDatasetWrapper: Wraps a dataset to use noisy labels from annotators.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict


class NoisyLabelDatasetWrapper(Dataset):
    """
    Wraps a dataset to use noisy labels from an AnnotatorManager.
    
    This wrapper intercepts __getitem__ calls and replaces true labels
    with noisy labels for samples that have been annotated.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        annotator_manager: Optional[object] = None,
        label_mapping: Optional[Dict[int, int]] = None
    ):
        """
        Initialize the wrapper.
        
        Args:
            dataset: The original dataset
            annotator_manager: AnnotatorManager instance (optional)
            label_mapping: Direct mapping from sample_idx to label (optional)
                          If both are provided, annotator_manager takes precedence
        """
        self.dataset = dataset
        self.annotator_manager = annotator_manager
        self.label_mapping = label_mapping or {}
        # Optional per-index HARD label override that takes precedence over the
        # annotator's noisy label (used by BayesianCoverage TRAIN_LABEL_MODE=
        # map_posterior to train on posterior-corrected labels). {global_idx: label}.
        self.training_label_override = {}

        # Get original targets for fallback
        if hasattr(dataset, 'targets'):
            self.original_targets = dataset.targets
        else:
            self.original_targets = None
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index: int):
        """
        Get item with potentially noisy label.
        
        Args:
            index: Index of the sample
            
        Returns:
            Tuple of (image, label) where label may be noisy
        """
        # Get the original item
        img, true_label = self.dataset[index]

        # A training-time relabel (e.g. map_posterior) wins over everything.
        if index in self.training_label_override:
            return img, self.training_label_override[index]

        # Determine which label to use
        if self.annotator_manager is not None:
            label = self.annotator_manager.get_label_for_sample(index, true_label)
        elif index in self.label_mapping:
            # Use direct label mapping
            label = self.label_mapping[index]
        else:
            # Use true label
            label = true_label
        
        return img, label
    
    def __getattr__(self, name):
        """Forward attribute access to the wrapped dataset."""
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(self.dataset, name)
    
    def update_label_mapping(self, new_mapping: Dict[int, int]):
        """Update the label mapping with new entries."""
        if self.label_mapping is None:
            self.label_mapping = {}
        self.label_mapping.update(new_mapping)
    
    def set_annotator_manager(self, annotator_manager):
        """Set or update the annotator manager."""
        self.annotator_manager = annotator_manager


class SoftTargetDatasetWrapper(Dataset):
    """
    Serves per-sample SOFT training targets (a [num_classes] float vector) instead
    of an integer label, for BayesianCoverage TRAIN_LABEL_MODE in
    {soft_posterior, reweighted}. `soft_targets` maps global index -> vector; any
    index not present falls back to a one-hot of the base label (from the wrapped
    dataset, which is itself the NoisyLabelDatasetWrapper serving noisy labels).

    The soft target flows through the existing (inputs, labels) DataLoader tuple
    as a 2-D float batch [B, num_classes]; train_epoch detects the extra dim and
    switches to a soft cross-entropy. reweighted encodes its per-sample weight by
    scaling a one-hot row (sum = weight), so the same soft-CE path handles it.
    """

    def __init__(self, dataset, soft_targets, num_classes):
        self.dataset = dataset
        self.soft_targets = soft_targets
        self.num_classes = int(num_classes)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, base_label = self.dataset[index]
        vec = self.soft_targets.get(index)
        if vec is None:
            vec = np.zeros(self.num_classes, dtype=np.float32)
            vec[int(base_label)] = 1.0
        return img, torch.as_tensor(np.asarray(vec), dtype=torch.float32)

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(self.dataset, name)
