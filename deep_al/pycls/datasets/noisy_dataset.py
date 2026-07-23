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
        noise_transition_matrix: Optional transition matrix.
            If used for asymmetric noise and it has diagonal mass, the matrix is
            interpreted as the full transition matrix P(noisy_label | clean_label)
            and is applied to every sample.
            If the matrix has zero diagonal, it is interpreted as conditional
            flip probabilities and is applied only to the selected noisy indices.
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
        self.noise_type = str(noise_type).lower()
        self.noise_rate = float(noise_rate)
        self.seed = int(seed or 0)
        self.save_dir = save_dir

        # Get clean targets from dataset. Prefer dataset.targets because this
        # wrapper should receive the clean base dataset.
        if hasattr(dataset, "targets"):
            self.clean_labels = np.array(dataset.targets, dtype=np.int64)
        elif hasattr(dataset, "clean_labels"):
            self.clean_labels = np.array(dataset.clean_labels, dtype=np.int64)
        else:
            raise ValueError("Dataset must have a 'targets' or 'clean_labels' attribute")

        # Infer number of classes if not provided.
        if num_classes is None:
            num_classes = int(len(np.unique(self.clean_labels)))
        self.num_classes = int(num_classes)

        # Initialize active labels.
        self.noisy_labels = self.clean_labels.copy()
        self.is_noisy = np.zeros_like(self.clean_labels, dtype=bool)
        self.targets = self.noisy_labels.tolist()

        # Apply noise if specified.
        if self.noise_type != "clean" and self.noise_rate > 0.0:
            self._apply_noise(noise_transition_matrix)

        self._sync_targets()

    def _sync_targets(self):
        """Expose noisy labels through .targets for downstream code."""
        self.noisy_labels = np.asarray(self.noisy_labels, dtype=np.int64)
        self.is_noisy = self.noisy_labels != self.clean_labels
        self.targets = self.noisy_labels.tolist()

    @staticmethod
    def _normalize_transition_matrix(matrix: np.ndarray, num_classes: int) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (num_classes, num_classes):
            raise ValueError(
                f"Transition matrix shape {matrix.shape} does not match num_classes {num_classes}"
            )

        row_sums = matrix.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            bad_rows = np.flatnonzero(row_sums.squeeze() <= 0).tolist()
            raise ValueError(f"Transition matrix contains empty rows: {bad_rows[:10]}")

        return matrix / row_sums

    def _apply_noise(self, noise_transition_matrix=None):
        """Apply the specified noise type to labels."""
        rng = np.random.default_rng(self.seed)
        n_samples = len(self.clean_labels)

        if self.noise_type == "sym":
            n_noisy = int(self.noise_rate * n_samples)
            noise_idx = rng.choice(n_samples, n_noisy, replace=False)
            self._add_symmetric_noise(noise_idx, rng)

        elif self.noise_type == "asym":
            self._add_asymmetric_noise(noise_transition_matrix, rng)

        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

        self._sync_targets()

    def _add_symmetric_noise(self, noise_idx, rng):
        """Add symmetric noise: flip selected labels uniformly to any other class."""
        all_labels = np.arange(self.num_classes)

        for idx in noise_idx:
            valid_labels = all_labels[all_labels != self.clean_labels[idx]]
            self.noisy_labels[idx] = rng.choice(valid_labels)

    def _add_asymmetric_noise(self, noise_transition_matrix, rng):
        """Add asymmetric noise using a transition matrix.

        Two cases are supported:

        1. Full transition matrix with diagonal mass, e.g. diag=0.8 and
           off-diagonal mass=0.2. In this case, sample every label from
           P(noisy_label | clean_label). This avoids the old bug where the code
           first selected 20% of samples and then applied a matrix that already
           encoded 20% noise, producing roughly 4% changed labels.

        2. Conditional flip matrix with zero diagonal. In this case, first select
           noise_rate * N samples, then sample their new labels from the
           row-normalized off-diagonal probabilities.
        """
        if noise_transition_matrix is None:
            raise ValueError("Asymmetric noise requires a noise_transition_matrix")

        transition = self._normalize_transition_matrix(noise_transition_matrix, self.num_classes)
        diag = np.diag(transition)

        has_diagonal_mass = bool(np.any(diag > 1e-12))

        if has_diagonal_mass:
            # Full transition matrix: apply to every sample.
            for idx, true_label in enumerate(self.clean_labels):
                self.noisy_labels[idx] = rng.choice(self.num_classes, p=transition[int(true_label)])
            return

        # Conditional flip matrix: select exactly noise_rate * N samples, then
        # sample from off-diagonal probabilities.
        n_samples = len(self.clean_labels)
        n_noisy = int(self.noise_rate * n_samples)
        noise_idx = rng.choice(n_samples, n_noisy, replace=False)

        for idx in noise_idx:
            true_label = int(self.clean_labels[idx])
            probs = transition[true_label].copy()
            probs[true_label] = 0.0
            s = probs.sum()
            if s <= 0:
                # Fallback: uniform flip to any other class.
                valid_labels = np.arange(self.num_classes)
                valid_labels = valid_labels[valid_labels != true_label]
                self.noisy_labels[idx] = rng.choice(valid_labels)
            else:
                probs = probs / s
                self.noisy_labels[idx] = rng.choice(self.num_classes, p=probs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """Get item with noisy label."""
        sample, _ = self.dataset[index]
        noisy_label = int(self.noisy_labels[index])
        return sample, noisy_label

    def get_clean_label(self, index):
        """Get the clean label for an index."""
        return int(self.clean_labels[index])

    def get_noisy_info(self, index):
        """Get information about label noise for an index."""
        return {
            "clean_label": int(self.clean_labels[index]),
            "noisy_label": int(self.noisy_labels[index]),
            "is_noisy": bool(self.is_noisy[index]),
        }

    def get_noise_statistics(self):
        """Get statistics about the noise in the dataset."""
        actual_noise_rate = float(np.mean(self.is_noisy.astype(int)))
        return {
            "noise_type": self.noise_type,
            "requested_noise_rate": self.noise_rate,
            "actual_noise_rate": actual_noise_rate,
            "num_noisy_samples": int(np.sum(self.is_noisy)),
            "total_samples": int(len(self.clean_labels)),
        }
