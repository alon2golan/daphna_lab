"""
NoisyLabelDatasetWrapper: Wraps a dataset to use noisy labels from annotators.
"""
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
        
        # Determine which label to use
        if self.annotator_manager is not None:
            # Use annotator manager to get label
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
