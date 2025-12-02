"""
Generic Noisy Dataset Wrapper
Adds label noise to any dataset while preserving the original clean labels.
"""
import numpy as np
from torch.utils.data import Dataset


class NoisyDataset(Dataset):
    """
    A wrapper that adds label noise to any dataset.
    
    Args:
        dataset: Base dataset instance (CIFAR10, CIFAR100, ImageNet, etc.)
        noise_type: Type of noise - 'sym' (symmetric), 'asym' (asymmetric), or 'clean'
        noise_rate: Noise rate between 0.0 and 1.0
        num_classes: Number of classes in the dataset
        seed: Random seed for reproducibility
        noise_transition_matrix: Optional custom transition matrix for asymmetric noise
        save_dir: Optional directory to save/load noisy labels
    """
    
    def __init__(
        self,
        dataset: Dataset,
        noise_type: str = "clean",
        noise_rate: float = 0.0,
        num_classes: int = None,
        seed: int = 0,
        noise_transition_matrix: np.ndarray = None,
        save_dir: str = None,
    ):
        self.dataset = dataset
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.seed = seed
        self.save_dir = save_dir
        
        # Get targets from dataset
        if hasattr(dataset, 'targets'):
            self.clean_labels = np.array(dataset.targets, dtype=np.int64)
        else:
            raise ValueError("Dataset must have a 'targets' attribute")
        
        # Infer number of classes if not provided
        if num_classes is None:
            num_classes = len(np.unique(self.clean_labels))
        self.num_classes = num_classes
        
        # Initialize noisy labels
        self.noisy_labels = self.clean_labels.copy()
        self.is_noisy = np.zeros_like(self.clean_labels, dtype=bool)
        
        # Apply noise if specified
        if noise_type != "clean" and noise_rate > 0.0:
            self._apply_noise(noise_transition_matrix)
    
    def _apply_noise(self, noise_transition_matrix=None):
        """Apply the specified noise type to labels."""
        np.random.seed(self.seed)
        n_samples = len(self.clean_labels)
        noise_idx = np.random.choice(
            n_samples,
            int(self.noise_rate * n_samples),
            replace=False
        )
        
        if self.noise_type == "sym":
            self._add_symmetric_noise(noise_idx)
        elif self.noise_type == "asym":
            self._add_asymmetric_noise(noise_idx, noise_transition_matrix)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        
        # Track which labels are noisy
        self.is_noisy = self.noisy_labels != self.clean_labels
    
    def _add_symmetric_noise(self, noise_idx):
        """Add symmetric noise: randomly flip labels to any other class."""
        all_labels = np.arange(self.num_classes)
        
        for idx in noise_idx:
            # Get all labels except the current one
            valid_labels = all_labels[all_labels != self.clean_labels[idx]]
            # Randomly pick one
            self.noisy_labels[idx] = np.random.choice(valid_labels)
    
    def _add_asymmetric_noise(self, noise_idx, noise_transition_matrix=None):
        """Add asymmetric noise using a transition matrix."""
        if noise_transition_matrix is None:
            raise ValueError("Asymmetric noise requires a noise_transition_matrix")
        
        if noise_transition_matrix.shape != (self.num_classes, self.num_classes):
            raise ValueError(
                f"Transition matrix shape {noise_transition_matrix.shape} "
                f"does not match num_classes {self.num_classes}"
            )
        
        for idx in noise_idx:
            true_label = self.clean_labels[idx]
            # Get the transition probabilities for this label
            transition_probs = noise_transition_matrix[true_label]
            # Sample new label according to transition probabilities
            self.noisy_labels[idx] = np.random.choice(
                self.num_classes,
                p=transition_probs
            )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        """Get item with noisy label."""
        sample, _ = self.dataset[index]
        noisy_label = self.noisy_labels[index]
        return sample, noisy_label
    
    def get_clean_label(self, index):
        """Get the clean label for an index."""
        return self.clean_labels[index]
    
    def get_noisy_info(self, index):
        """Get information about label noise for an index."""
        return {
            'clean_label': self.clean_labels[index],
            'noisy_label': self.noisy_labels[index],
            'is_noisy': self.is_noisy[index]
        }
    
    def get_noise_statistics(self):
        """Get statistics about the noise in the dataset."""
        actual_noise_rate = np.mean(self.is_noisy.astype(int))
        return {
            'noise_type': self.noise_type,
            'requested_noise_rate': self.noise_rate,
            'actual_noise_rate': actual_noise_rate,
            'num_noisy_samples': np.sum(self.is_noisy),
            'total_samples': len(self.clean_labels)
        }
