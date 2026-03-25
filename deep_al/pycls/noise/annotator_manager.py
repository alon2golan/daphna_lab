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
        self.annotated_labels = {}  # {sample_idx: annotated_label}
        self.cumulative_cost = 0.0
        
        # Annotator statistics (populated by _compute_annotator_stats)
        self.annotator_accuracies: List[float] = []
        self.annotator_costs: List[float] = []
        self.best_annotator_id: Optional[int] = None
        
        # Compute annotator statistics
        self._compute_annotator_stats()
    
    def _compute_annotator_stats(self):
        """Compute and cache annotator statistics."""
        self.annotator_accuracies = [ann.accuracy for ann in self.annotators]
        self.annotator_costs = [ann.cost for ann in self.annotators]
        self.best_annotator_id = None
        
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
    
    def select_annotator(self) -> int:
        """
        Select an annotator based on the configured strategy.

        Returns:
            int: Index of the selected annotator
        """
        if self.selection_strategy == "random":
            return self.rng.integers(0, len(self.annotators))
        elif self.selection_strategy in ["most_accurate", "least_cost", "cost_accuracy_ratio", "forced"]:
            return self.best_annotator_id
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

    def force_annotator(self, annotator_id: int):
        """Force the manager to always select a specific annotator.
        
        Args:
            annotator_id: Index into self.annotators to always use.
        """
        self.selection_strategy = "forced"
        self.best_annotator_id = annotator_id

    def plan_annotation_round(self, budget: float) -> tuple:
        """
        Select the annotator for this round and compute how many samples
        can be annotated within the given budget.

        Returns both values so the caller can pass the same annotator_id to
        annotate_samples(), keeping selection and annotation in sync.

        Args:
            budget: Total annotation budget for this round.

        Returns:
            (num_samples, annotator_id): number of samples affordable and the
            annotator index selected for this round.
        """
        annotator_id = self.select_annotator()
        cost_per_sample = self.annotators[annotator_id].cost

        if cost_per_sample <= 0:
            # Free annotator (oracle): budget directly equals number of samples
            return int(budget), annotator_id
        return int(budget / cost_per_sample), annotator_id

    def annotate_samples(
        self,
        sample_indices: np.ndarray,
        annotator_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Annotate a set of samples using a single annotator for the entire batch.

        The annotator is chosen once per call (either the provided annotator_id or
        the one returned by select_annotator()), so all samples in this batch are
        labelled by the same person.  The caller is responsible for sizing the
        sample list (e.g. via plan_annotation_round) before calling this method.

        Args:
            sample_indices: Array of sample indices to annotate
            annotator_id: Specific annotator to use (None = use selection strategy)

        Returns:
            Dictionary with:
                - 'annotated_samples': List of sample indices that were annotated
                - 'annotator_assignments': Dict mapping sample_idx to annotator_id
                - 'annotated_labels': Dict mapping sample_idx to annotated label
                - 'cost_spent': Total cost spent
        """
        # Pick one annotator for the whole batch
        if annotator_id is None:
            annotator_id = self.select_annotator()

        cost_per_sample = self.annotators[annotator_id].cost

        annotated_samples = []
        local_assignments = {}
        local_labels = {}
        cost_spent = 0.0

        for idx in sample_indices:
            noisy_label = self.annotators[annotator_id].get_label(idx)

            self.annotator_assignments[int(idx)] = int(annotator_id)
            self.annotated_labels[int(idx)] = int(noisy_label)
            local_assignments[int(idx)] = int(annotator_id)
            local_labels[int(idx)] = int(noisy_label)

            annotated_samples.append(int(idx))
            cost_spent += cost_per_sample

        self.cumulative_cost += cost_spent

        return {
            'annotated_samples': annotated_samples,
            'annotator_assignments': local_assignments,
            'annotated_labels': local_labels,
            'cost_spent': cost_spent,
        }
    
    def get_label_for_sample(self, sample_idx: int, true_label: int) -> int:
        """
        Get the label to use for a sample during training.
        
        Args:
            sample_idx: Index of the sample
            true_label: True label from dataset
            
        Returns:
            int: Label to use (noisy if annotated, true if oracle/not annotated)
        """
        if sample_idx in self.annotated_labels:
            return self.annotated_labels[sample_idx]
        else:
            # Oracle or not yet annotated - use true label
            return true_label
    
    def get_annotator_for_sample(self, sample_idx: int):
        """Get the annotator ID for a sample, or None if unassigned."""
        return self.annotator_assignments.get(sample_idx, None)
    
    def save_state(self, episode_dir: str):
        """Save annotator assignments and labels to disk."""
        if not os.path.exists(episode_dir):
            os.makedirs(episode_dir)
        
        # Save annotator assignments
        assignments_file = os.path.join(episode_dir, 'annotator_assignments.json')
        with open(assignments_file, 'w') as f:
            json.dump(self.annotator_assignments, f, indent=2)
        
        # Save annotated labels
        labels_file = os.path.join(episode_dir, 'annotated_labels.json')
        with open(labels_file, 'w') as f:
            json.dump(self.annotated_labels, f, indent=2)
        
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
        
        labels_file = os.path.join(episode_dir, 'annotated_labels.json')
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                loaded = json.load(f)
                self.annotated_labels = {int(k): v for k, v in loaded.items()}
        
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
