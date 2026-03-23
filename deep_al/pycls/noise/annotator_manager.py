"""
AnnotatorManager: Manages multiple annotators and selection strategies for active learning.
"""
import numpy as np
import os
import json
from typing import List, Dict, Optional, Any
from .annotator import Annotator


class AnnotatorManager:
    """Manages multiple annotators and handles simple selection strategies."""
    
    # Special annotator ID for oracle/unknown annotator
    ORACLE_ANNOTATOR_ID = -1
    
    def __init__(
        self,
        annotators: List[Annotator],
        selection_strategy: str = "random",
        seed: int = 0,
        save_dir: Optional[str] = None
    ):
        """
        Initialize the AnnotatorManager.
        
        Args:
            annotators: List of Annotator objects
            selection_strategy: Strategy for selecting annotators
                - "random": Randomly select among all annotators
                - "most_accurate": Always select the most accurate annotator
                - "least_cost": Always select the cheapest annotator
                - "cost_accuracy_ratio": Select based on best accuracy/cost ratio
            seed: Random seed for reproducibility
            save_dir: Directory to save annotator assignments and labels
        """
        self.annotators = annotators
        self.selection_strategy = selection_strategy
        self.seed = seed
        self.save_dir = save_dir
        self.rng = np.random.default_rng(seed)
        
        # Tracking structures
        self.annotator_assignments = {}  # {sample_idx: annotator_id}
        self.noisy_labels = {}  # {sample_idx: noisy_label}
        self.cumulative_cost = 0.0
        
        # Compute annotator statistics
        self._compute_annotator_stats()
    
    def _compute_annotator_stats(self):
        """Compute and cache annotator statistics."""
        self.annotator_accuracies = [ann.accuracy for ann in self.annotators]
        self.annotator_costs = [ann.cost for ann in self.annotators]
        
        # For strategies that need it
        if self.selection_strategy == "most_accurate":
            self.best_annotator_id = int(np.argmax(self.annotator_accuracies))
        elif self.selection_strategy == "least_cost":
            self.best_annotator_id = int(np.argmin(self.annotator_costs))
        elif self.selection_strategy == "cost_accuracy_ratio":
            # Select annotator with best accuracy/cost ratio
            ratios = [acc / max(cost, 1e-6) for acc, cost in 
                     zip(self.annotator_accuracies, self.annotator_costs)]
            self.best_annotator_id = int(np.argmax(ratios))
    
    def select_annotator(self, sample_idx: Optional[int] = None) -> int:
        """
        Select an annotator based on the configured strategy.
        
        Args:
            sample_idx: Index of the sample (can be used for sample-specific strategies)
            
        Returns:
            int: Index of the selected annotator
        """
        if self.selection_strategy == "random":
            return self.rng.integers(0, len(self.annotators))
        elif self.selection_strategy in ["most_accurate", "least_cost", "cost_accuracy_ratio"]:
            return self.best_annotator_id
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

    def compute_num_samples_from_budget(self, budget: float) -> int:
        """
        Compute how many samples can be annotated with the given budget,
        using the annotator that would be selected by the current strategy.

        For a uniform per-sample cost c, this is floor(budget / c).
        If the selected annotator has cost 0 (oracle), returns the budget
        value itself as an integer (caller should pass a meaningful fallback).

        Args:
            budget: Total annotation budget for this round.

        Returns:
            int: Number of samples affordable with this budget.
        """
        annotator_id = self.select_annotator()
        cost_per_sample = self.annotators[annotator_id].cost
        if cost_per_sample <= 0:
            # Free annotator (oracle): budget directly equals number of samples
            return int(budget)
        return int(budget / cost_per_sample)

    def annotate_samples(
        self,
        sample_indices: np.ndarray,
        budget: Optional[float] = None,
        annotator_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Annotate a set of samples, respecting budget constraints.
        
        Args:
            sample_indices: Array of sample indices to annotate
            budget: Maximum budget to spend (None = no limit)
            annotator_id: Specific annotator to use (None = use selection strategy)
            
        Returns:
            Dictionary with:
                - 'annotated_samples': List of sample indices that were annotated
                - 'annotator_assignments': Dict mapping sample_idx to annotator_id
                - 'noisy_labels': Dict mapping sample_idx to noisy label
                - 'cost_spent': Total cost spent
                - 'samples_skipped': Number of samples skipped due to budget
        """
        annotated_samples = []
        local_assignments = {}
        local_labels = {}
        cost_spent = 0.0
        samples_skipped = 0
        
        for idx in sample_indices:
            # Select annotator
            if annotator_id is not None:
                selected_annotator_id = annotator_id
            else:
                selected_annotator_id = self.select_annotator(idx)
            
            # Check budget
            annotator_cost = self.annotators[selected_annotator_id].cost
            if budget is not None and cost_spent + annotator_cost > budget:
                samples_skipped += 1
                continue
            
            # Get label from annotator
            noisy_label = self.annotators[selected_annotator_id].get_label(idx)
            
            # Update tracking
            self.annotator_assignments[int(idx)] = int(selected_annotator_id)
            self.noisy_labels[int(idx)] = int(noisy_label)
            local_assignments[int(idx)] = int(selected_annotator_id)
            local_labels[int(idx)] = int(noisy_label)
            
            annotated_samples.append(int(idx))
            cost_spent += annotator_cost
        
        self.cumulative_cost += cost_spent
        
        return {
            'annotated_samples': annotated_samples,
            'annotator_assignments': local_assignments,
            'noisy_labels': local_labels,
            'cost_spent': cost_spent,
            'samples_skipped': samples_skipped
        }
    
    def annotate_with_oracle(self, sample_indices: np.ndarray):
        """
        Mark samples as annotated by oracle (no noise).
        This is used for initial labeled set or validation set.
        
        Args:
            sample_indices: Array of sample indices
        """
        for idx in sample_indices:
            self.annotator_assignments[int(idx)] = self.ORACLE_ANNOTATOR_ID
            # For oracle, we don't set noisy_labels (will use true labels)
    
    def get_label_for_sample(self, sample_idx: int, true_label: int) -> int:
        """
        Get the label to use for a sample during training.
        
        Args:
            sample_idx: Index of the sample
            true_label: True label from dataset
            
        Returns:
            int: Label to use (noisy if annotated, true if oracle/not annotated)
        """
        if sample_idx in self.noisy_labels:
            return self.noisy_labels[sample_idx]
        else:
            # Oracle or not yet annotated - use true label
            return true_label
    
    def get_annotator_for_sample(self, sample_idx: int) -> int:
        """Get the annotator ID for a sample."""
        return self.annotator_assignments.get(sample_idx, self.ORACLE_ANNOTATOR_ID)
    
    def save_state(self, episode_dir: str):
        """Save annotator assignments and labels to disk."""
        if not os.path.exists(episode_dir):
            os.makedirs(episode_dir)
        
        # Save annotator assignments
        assignments_file = os.path.join(episode_dir, 'annotator_assignments.json')
        with open(assignments_file, 'w') as f:
            json.dump(self.annotator_assignments, f, indent=2)
        
        # Save noisy labels
        labels_file = os.path.join(episode_dir, 'noisy_labels.json')
        with open(labels_file, 'w') as f:
            json.dump(self.noisy_labels, f, indent=2)
        
        # Save cumulative cost
        cost_file = os.path.join(episode_dir, 'cumulative_cost.txt')
        with open(cost_file, 'w') as f:
            f.write(f"{self.cumulative_cost}\n")
    
    def load_state(self, episode_dir: str):
        """Load annotator assignments and labels from disk."""
        assignments_file = os.path.join(episode_dir, 'annotator_assignments.json')
        if os.path.exists(assignments_file):
            with open(assignments_file, 'r') as f:
                loaded = json.load(f)
                # Convert string keys back to int
                self.annotator_assignments = {int(k): v for k, v in loaded.items()}
        
        labels_file = os.path.join(episode_dir, 'noisy_labels.json')
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                loaded = json.load(f)
                self.noisy_labels = {int(k): v for k, v in loaded.items()}
        
        cost_file = os.path.join(episode_dir, 'cumulative_cost.txt')
        if os.path.exists(cost_file):
            with open(cost_file, 'r') as f:
                self.cumulative_cost = float(f.read().strip())
    
    def get_statistics(self) -> Dict:
        """Get statistics about annotations."""
        annotator_counts = {}
        for ann_id in self.annotator_assignments.values():
            annotator_counts[ann_id] = annotator_counts.get(ann_id, 0) + 1
        
        return {
            'total_annotated': len(self.annotator_assignments),
            'total_cost': self.cumulative_cost,
            'annotator_counts': annotator_counts,
            'annotator_accuracies': self.annotator_accuracies,
            'annotator_costs': self.annotator_costs,
            'selection_strategy': self.selection_strategy
        }
