"""
Generic Noisy Dataset Wrapper
Adds label noise to any dataset while preserving the original clean labels.
"""
import numpy as np
from torch.utils.data import Dataset
from typing import Optional

# Offset so the annotator's noise RNG never collides with cfg.RNG_SEED (the
# data-split seed). When they coincided, np.random.choice for the noise indices
# reproduced the exact prefix of the makeLUVSets shuffle permutation, so the
# corrupted samples landed disproportionately in the ProbCover selection pool
# (realized lset noise ~45% at nominal nu=0.30). Decoupling restores the
# nominal rate.
NOISE_SEED_OFFSET = 10_000


class Annotator:
    """A Annotator wrapper that adds label noise to any dataset."""
    
    def __init__(
        self,
        dataset: Dataset,
        noise_type: str = "clean",
        noise_rate: float = 0.0,
        cost: float = 0.0,
        num_classes: Optional[int] = None,
        seed: int = 0,
        noise_transition_matrix: Optional[np.ndarray] = None,
        save_dir: Optional[str] = None,
    ):
        self.dataset = dataset
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.cost = cost
        self.seed = seed
        self.save_dir = save_dir
        # Retained so consumers (e.g. BayesianCoverage Step A) can reconstruct the
        # effective confusion matrix for asymmetric annotators.
        self.noise_transition_matrix = noise_transition_matrix
        
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
        # Dedicated RNG, offset away from cfg.RNG_SEED, and isolated from the
        # global np.random stream (see NOISE_SEED_OFFSET note above).
        self._noise_rng = np.random.RandomState(self.seed + NOISE_SEED_OFFSET)
        n_samples = len(self.clean_labels)
        noise_idx = self._noise_rng.choice(
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
            self.noisy_labels[idx] = self._noise_rng.choice(valid_labels)
    
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
            self.noisy_labels[idx] = self._noise_rng.choice(
                self.num_classes,
                p=transition_probs
            )
    
    def __len__(self):
        return len(self.dataset)
    
    def get_annotation(self, index):
        """Get item with noisy label."""
        sample, _ = self.dataset[index]
        noisy_label = self.noisy_labels[index]
        clean_label = self.clean_labels[index]
        return sample, noisy_label, clean_label
    
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
    
    def get_label(self, index):
        """Get the noisy label for a given sample index.
        
        Args:
            index: int, Index of the sample in the dataset
            
        Returns:
            int, The noisy label for the sample
        """
        return int(self.noisy_labels[index])
    
    @property
    def accuracy(self):
        """Get the accuracy of this annotator (1 - noise_rate)."""
        return 1.0 - self.noise_rate

    @property
    def confusion_matrix(self):
        """
        Effective confusion matrix M[k, j] = P(observed = j | true = k), i.e. the
        probability this annotator reports label j for a sample whose true label
        is k. Reflects how noise is actually injected in _apply_noise: a fraction
        `noise_rate` of samples is resampled, the rest keep the true label, so

            M = (1 - noise_rate) * I  +  noise_rate * R,

        where R[k, ·] is the resampling distribution for class k:
          * clean / noise_rate == 0 -> identity (R unused),
          * 'sym'  -> uniform over the OTHER C-1 classes (R[k,k] = 0),
          * 'asym' -> the provided noise_transition_matrix rows.

        Returns a (num_classes, num_classes) float64 numpy array with rows that
        sum to 1.
        """
        C = self.num_classes
        if self.noise_type == "clean" or self.noise_rate <= 0.0:
            return np.eye(C, dtype=np.float64)

        nr = float(self.noise_rate)
        if self.noise_type == "sym":
            R = np.full((C, C), 1.0 / (C - 1), dtype=np.float64)
            np.fill_diagonal(R, 0.0)
        elif self.noise_type == "asym":
            if self.noise_transition_matrix is None:
                raise ValueError("asym annotator has no noise_transition_matrix "
                                 "to build a confusion matrix from")
            R = np.asarray(self.noise_transition_matrix, dtype=np.float64)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

        M = (1.0 - nr) * np.eye(C, dtype=np.float64) + nr * R
        return M
