"""
Veracity-Based Annotator Selection Strategy

This module implements a greedy strategy that balances annotation quality vs quantity
by maintaining a kernel-based veracity vector.

Veracity Definition:
    v_j = max_{i ∈ L} [(1 - noise_rate(a_i)) * K_ij]
    
Where:
    - L = labeled set
    - a_i = annotator who labeled example i
    - K_ij = kernel similarity between examples i and j
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import cdist


class VeracityBasedAnnotatorSelector:
    """
    Implements kernel-based veracity annotator selection strategy.
    
    Veracity is computed based on similarity to labeled samples and their
    annotator quality: v_j = max_i[(1 - noise_rate(a_i)) * K_ij]
    """
    
    def __init__(
        self,
        annotators: List,
        oracle_id: int = 0,
        kernel_type: str = 'rbf',
        kernel_param: float = 1.0,
        seed: int = 0
    ):
        """
        Initialize the veracity-based selector.
        
        Args:
            annotators: List of Annotator objects (first must be oracle)
            oracle_id: Index of the oracle annotator (default: 0)
            kernel_type: Type of kernel function
                - 'rbf': RBF kernel (MaxHerding style)
                - 'delta_ball': Delta-ball kernel (ProbCover style)
            kernel_param: Kernel parameter
                - For 'rbf': sigma (bandwidth)
                - For 'delta_ball': delta (radius)
            seed: Random seed
        """
        self.annotators = annotators
        self.oracle_id = oracle_id
        self.kernel_type = kernel_type
        self.kernel_param = kernel_param
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Verify oracle is correctly configured
        assert self.annotators[oracle_id].noise_rate == 0.0, \
            f"Annotator {oracle_id} should be oracle (noise_rate=0.0)"
        
        # Labeled set tracking
        self.labeled_indices = []  # List of labeled sample indices
        self.labeled_features = None  # Features of labeled samples (N_labeled x D)
        self.labeled_annotators = {}  # {sample_idx: annotator_id}
        
        # Unlabeled set features
        self.unlabeled_features = None  # Features of unlabeled samples (N_unlabeled x D)
        self.unlabeled_indices = []  # List of unlabeled sample indices
        
        # Veracity scores
        self.veracity_scores = {}  # {sample_idx: veracity_score}
        
        # Precompute annotator statistics
        self.annotator_accuracies = [ann.accuracy for ann in self.annotators]
        self.annotator_costs = [ann.cost for ann in self.annotators]
    
    def compute_kernel(self, features_i: np.ndarray, features_j: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix between two sets of features.
        
        Args:
            features_i: (N_i x D) array of features
            features_j: (N_j x D) array of features
            
        Returns:
            K: (N_i x N_j) kernel matrix where K[i,j] = kernel(i, j)
        """
        # Compute pairwise distances
        # cdist computes distances between rows of features_i and features_j
        distances = cdist(features_i, features_j, metric='euclidean')
        
        if self.kernel_type == 'rbf':
            # RBF kernel: K_ij = exp(-||x_i - x_j||^2 / (2 * sigma^2))
            sigma = self.kernel_param
            K = np.exp(-distances**2 / (2 * sigma**2))
        
        elif self.kernel_type == 'delta_ball':
            # Delta-ball kernel: K_ij = 1 if ||x_i - x_j|| < delta, else 0
            delta = self.kernel_param
            K = (distances < delta).astype(float)
        
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        return K
    
    def initialize_features(
        self,
        lSet: np.ndarray,
        uSet: np.ndarray,
        features: np.ndarray,
        labeled_annotators: Optional[Dict[int, int]] = None
    ):
        """
        Initialize features for labeled and unlabeled sets.
        
        Args:
            lSet: Array of labeled sample indices
            uSet: Array of unlabeled sample indices
            features: (N_total x D) array of features for all samples
            labeled_annotators: Dict mapping labeled sample idx to annotator id
                               If None, assumes all labeled by oracle
        """
        self.labeled_indices = list(lSet)
        self.unlabeled_indices = list(uSet)
        
        # Extract features
        self.labeled_features = features[lSet]
        self.unlabeled_features = features[uSet]
        
        # Track annotators for labeled samples
        if labeled_annotators is not None:
            self.labeled_annotators = labeled_annotators.copy()
        else:
            # Default: all labeled by oracle
            self.labeled_annotators = {int(idx): self.oracle_id for idx in lSet}
        
        # Compute initial veracity scores
        self._compute_veracity_all()
    
    def _compute_veracity_all(self):
        """
        Compute veracity for all unlabeled samples.
        
        v_j = max_{i in L} [(1 - noise_rate(a_i)) * K_ij]
        """
        if len(self.labeled_indices) == 0:
            # No labeled samples yet - veracity is 0
            for idx in self.unlabeled_indices:
                self.veracity_scores[int(idx)] = 0.0
            return
        
        # Compute kernel matrix: (N_labeled x N_unlabeled)
        K = self.compute_kernel(self.labeled_features, self.unlabeled_features)
        
        # For each labeled sample i, compute (1 - noise_rate(a_i))
        labeled_accuracies = np.array([
            self.annotator_accuracies[self.labeled_annotators[int(idx)]]
            for idx in self.labeled_indices
        ])  # Shape: (N_labeled,)
        
        # Broadcast: (N_labeled, 1) * (N_labeled, N_unlabeled) = (N_labeled, N_unlabeled)
        weighted_kernel = labeled_accuracies[:, np.newaxis] * K
        
        # Take max over labeled samples for each unlabeled sample
        veracity_values = np.max(weighted_kernel, axis=0)  # Shape: (N_unlabeled,)
        
        # Store in dictionary
        for idx, veracity in zip(self.unlabeled_indices, veracity_values):
            self.veracity_scores[int(idx)] = float(veracity)
        # Store in dictionary
        for idx, veracity in zip(self.unlabeled_indices, veracity_values):
            self.veracity_scores[int(idx)] = float(veracity)
    
    def compute_veracity_increase_kernel_based(
        self,
        sample_indices: np.ndarray,
        annotator_id: int,
        features: np.ndarray
    ) -> float:
        """
        Compute expected veracity increase using kernel-based definition.
        
        Args:
            sample_indices: Samples to be annotated
            annotator_id: ID of the annotator
            features: Features for the samples
            
        Returns:
            float: Total expected veracity increase
        """
        if len(self.unlabeled_indices) == 0:
            return 0.0
        
        annotator_accuracy = self.annotator_accuracies[annotator_id]
        
        # Compute kernel between candidate samples and unlabeled samples
        # features: (N_candidates x D)
        # unlabeled_features: (N_unlabeled x D)
        K = self.compute_kernel(features, self.unlabeled_features)  # (N_candidates x N_unlabeled)
        
        # For each unlabeled sample j, compute potential new veracity
        # new_v_j = max(old_v_j, max_i[accuracy * K_ij])
        weighted_kernel = annotator_accuracy * K  # (N_candidates x N_unlabeled)
        max_new_contribution = np.max(weighted_kernel, axis=0)  # (N_unlabeled,)
        
        # Current veracity for unlabeled samples
        current_veracity = np.array([
            self.veracity_scores.get(int(idx), 0.0)
            for idx in self.unlabeled_indices
        ])
        
        # New veracity would be max of current and new contribution
        new_veracity = np.maximum(current_veracity, max_new_contribution)
        
        # Total veracity increase
        veracity_increase = np.sum(new_veracity - current_veracity)
        
        return float(veracity_increase)
    
    def select_annotator_and_samples(
        self,
        candidate_samples: np.ndarray,
        candidate_features: np.ndarray,
        budget: float,
        iteration_budget: Optional[float] = None
    ) -> Tuple[int, np.ndarray]:
        """
        Select annotator and samples that maximize veracity increase.
        
        Uses kernel-based veracity: v_j = max_i[(1 - noise_rate(a_i)) * K_ij]
        
        Args:
            candidate_samples: Array of candidate sample indices from AL
            candidate_features: (N_candidates x D) features for candidates
            budget: Remaining total budget
            iteration_budget: Budget for this iteration (if None, uses remaining budget)
            
        Returns:
            Tuple of (selected_annotator_id, selected_sample_indices)
        """
        if iteration_budget is None:
            iteration_budget = budget
        
        if len(candidate_samples) == 0:
            raise ValueError("No candidate samples available")
        
        # Sort candidates by current veracity (prioritize low veracity = high uncertainty)
        veracity_values = np.array([
            self.veracity_scores.get(int(idx), 0.0)
            for idx in candidate_samples
        ])
        sorted_order = np.argsort(veracity_values)  # Ascending order (low veracity first)
        sorted_indices = candidate_samples[sorted_order]
        sorted_features = candidate_features[sorted_order]
        
        best_option = None
        best_veracity_increase = -np.inf
        best_samples = None
        best_features = None
        
        # Evaluate each annotator option
        for ann_id, annotator in enumerate(self.annotators):
            cost = self.annotator_costs[ann_id]
            
            # Check if we can afford at least one sample
            if cost > iteration_budget:
                continue
            
            # How many samples can we get with this annotator?
            n_samples = min(
                int(iteration_budget / cost),
                len(candidate_samples)
            )
            
            if n_samples == 0:
                continue
            
            # Select n samples with lowest veracity
            selected_samples = sorted_indices[:n_samples]
            selected_features = sorted_features[:n_samples]
            
            # Compute veracity increase for this option using kernel-based method
            veracity_increase = self.compute_veracity_increase_kernel_based(
                selected_samples,
                ann_id,
                selected_features
            )
            
            # Normalize by cost for fair comparison
            veracity_per_cost = veracity_increase / (n_samples * cost) if (n_samples * cost) > 0 else 0
            
            print(f"  Option: Annotator {ann_id} (acc={self.annotator_accuracies[ann_id]:.2f}, cost={cost})")
            print(f"    -> {n_samples} samples, veracity_increase={veracity_increase:.4f}, per_cost={veracity_per_cost:.4f}")
            
            # Track best option
            if veracity_increase > best_veracity_increase:
                best_veracity_increase = veracity_increase
                best_option = ann_id
                best_samples = selected_samples
                best_features = selected_features
        
        if best_option is None:
            raise ValueError("No valid annotator option found within budget")
        
        print(f"  -> Selected: Annotator {best_option}, {len(best_samples)} samples, veracity_increase={best_veracity_increase:.4f}")
        
        return best_option, best_samples
    
    def update_veracity_after_annotation(
        self,
        sample_indices: np.ndarray,
        sample_features: np.ndarray,
        annotator_id: int
    ):
        """
        Update state after samples have been annotated.
        
        Args:
            sample_indices: Samples that were annotated
            sample_features: (N_annotated x D) features for annotated samples
            annotator_id: ID of the annotator used
        """
        # Add to labeled set
        for idx in sample_indices:
            idx_int = int(idx)
            self.labeled_indices.append(idx_int)
            self.labeled_annotators[idx_int] = annotator_id
            
            # Remove from unlabeled
            if idx_int in self.unlabeled_indices:
                unlabeled_pos = self.unlabeled_indices.index(idx_int)
                self.unlabeled_indices.pop(unlabeled_pos)
                # Remove corresponding feature row
                self.unlabeled_features = np.delete(self.unlabeled_features, unlabeled_pos, axis=0)
        
        # Append new features to labeled_features
        if self.labeled_features is None or len(self.labeled_features) == 0:
            self.labeled_features = sample_features
        else:
            self.labeled_features = np.vstack([self.labeled_features, sample_features])
        
        # Recompute veracity for all remaining unlabeled samples
        self._compute_veracity_all()
    
    def get_veracity_statistics(self) -> Dict:
        """Get statistics about current veracity scores."""
        if len(self.veracity_scores) == 0:
            return {
                'mean_veracity': 0.0,
                'min_veracity': 0.0,
                'max_veracity': 0.0,
                'num_samples': 0
            }
        
        scores = list(self.veracity_scores.values())
        return {
            'mean_veracity': float(np.mean(scores)),
            'min_veracity': float(np.min(scores)),
            'max_veracity': float(np.max(scores)),
            'median_veracity': float(np.median(scores)),
            'num_samples': len(scores)
        }
