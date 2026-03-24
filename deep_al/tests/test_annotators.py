"""
Unit tests for noisy annotator integration in active learning.

Tests:
1. Annotator creation and noise injection
2. AnnotatorManager selection strategies
3. MISP with annotator-weighted veracity
4. NoisyLabelDatasetWrapper label propagation
5. Backward compatibility with non-annotator AL algorithms
"""

import unittest
import numpy as np
import torch
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add repo root to path
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from deep_al.pycls.noise.annotator import Annotator
from deep_al.pycls.noise.annotator_manager import AnnotatorManager
from deep_al.pycls.noise.noisy_label_wrapper import NoisyLabelDatasetWrapper


class AttrDict(dict):
    """Dict that supports both dict['key'] and dict.key access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value


class MockDataset:
    """Mock dataset for testing."""
    def __init__(self, size=100, num_classes=10):
        self.targets = np.random.randint(0, num_classes, size).tolist()
        self.data = np.random.randn(size, 32, 32, 3).astype(np.float32)
        self.features = np.random.randn(size, 128).astype(np.float32)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class TestAnnotator(unittest.TestCase):
    """Test Annotator class."""
    
    def setUp(self):
        self.dataset = MockDataset(size=100, num_classes=10)
        self.seed = 42
    
    def test_oracle_annotator(self):
        """Test oracle annotator returns clean labels."""
        annotator = Annotator(
            dataset=self.dataset,
            noise_type="clean",
            noise_rate=0.0,
            cost=1.0,
            seed=self.seed
        )
        
        self.assertEqual(annotator.accuracy, 1.0)
        self.assertEqual(annotator.cost, 1.0)
        
        # Check all labels are clean
        for idx in range(len(self.dataset)):
            self.assertEqual(
                annotator.get_label(idx),
                self.dataset.targets[idx]
            )
    
    def test_noisy_annotator(self):
        """Test noisy annotator injects noise."""
        noise_rate = 0.3
        annotator = Annotator(
            dataset=self.dataset,
            noise_type="sym",
            noise_rate=noise_rate,
            cost=0.5,
            seed=self.seed
        )
        
        self.assertAlmostEqual(annotator.accuracy, 1.0 - noise_rate, places=5)
        self.assertEqual(annotator.cost, 0.5)
        
        # Count noisy labels
        noisy_count = 0
        for idx in range(len(self.dataset)):
            if annotator.get_label(idx) != self.dataset.targets[idx]:
                noisy_count += 1
        
        # Should be approximately noise_rate proportion
        actual_noise_rate = noisy_count / len(self.dataset)
        self.assertAlmostEqual(actual_noise_rate, noise_rate, delta=0.1)
    
    def test_annotator_cost(self):
        """Test annotator cost tracking."""
        annotator = Annotator(
            dataset=self.dataset,
            noise_type="clean",
            noise_rate=0.0,
            cost=0.25,
            seed=self.seed
        )
        
        self.assertEqual(annotator.cost, 0.25)


class TestAnnotatorManager(unittest.TestCase):
    """Test AnnotatorManager class."""
    
    def setUp(self):
        self.dataset = MockDataset(size=100, num_classes=10)
        
        # Create 3 annotators: oracle + 2 noisy
        self.annotators = [
            Annotator(self.dataset, "sym", 0.1, 0.3, seed=42),
            Annotator(self.dataset, "sym", 0.2, 0.5, seed=43),
            Annotator(self.dataset, "sym", 0.3, 0.7, seed=44),
        ]
        
        self.manager = AnnotatorManager(
            annotators=self.annotators,
            selection_strategy="random",
            seed=42
        )
    
    def test_manager_initialization(self):
        """Test manager initializes correctly."""
        self.assertEqual(len(self.manager.annotators), 3)
        self.assertEqual(self.manager.cumulative_cost, 0.0)
        self.assertEqual(len(self.manager.annotator_assignments), 0)
    
    def test_annotate_with_oracle(self):
        """Test oracle annotation."""
        samples = np.array([0, 1, 2, 3, 4])
        self.manager.annotate_with_oracle(samples)
        
        for idx in samples:
            self.assertEqual(
                self.manager.get_annotator_for_sample(idx),
                AnnotatorManager.ORACLE_ANNOTATOR_ID
            )
    
    def test_random_selection(self):
        """Test random annotator selection."""
        manager = AnnotatorManager(
            annotators=self.annotators,
            selection_strategy="random",
            seed=42
        )
        
        samples = np.arange(20)
        result = manager.annotate_samples(samples)
        
        self.assertEqual(len(result['annotated_samples']), 20)
        
        # Check that different annotators were used
        annotator_ids = [manager.get_annotator_for_sample(idx) for idx in samples]
        unique_annotators = set(annotator_ids)
        self.assertGreater(len(unique_annotators), 1)  # Should use multiple annotators
    
    def test_least_cost_selection(self):
        """Test least cost annotator selection."""
        manager = AnnotatorManager(
            annotators=self.annotators,
            selection_strategy="least_cost",
            seed=42
        )
        
        samples = np.arange(10)
        result = manager.annotate_samples(samples)

        # All should be annotated by cheapest annotator (id=0, cost=0.3)
        for idx in samples:
            annotator_id = manager.get_annotator_for_sample(idx)
            self.assertEqual(annotator_id, 0)
    
    def test_most_accurate_selection(self):
        """Test most accurate annotator selection."""
        manager = AnnotatorManager(
            annotators=self.annotators,
            selection_strategy="most_accurate",
            seed=42
        )
        
        samples = np.arange(10)
        result = manager.annotate_samples(samples)

        # All should be annotated by most accurate (oracle, id=0)
        for idx in samples:
            annotator_id = manager.get_annotator_for_sample(idx)
            self.assertEqual(annotator_id, 0)
    
    def test_plan_annotation_round(self):
        """Test plan_annotation_round returns correct sample count and annotator."""
        manager = AnnotatorManager(
            annotators=self.annotators,
            selection_strategy="least_cost",
            seed=42
        )
        budget = 10.0
        num_samples, annotator_id = manager.plan_annotation_round(budget)

        cost_per_sample = self.annotators[annotator_id].cost
        expected = int(budget / cost_per_sample) if cost_per_sample > 0 else int(budget)
        self.assertEqual(num_samples, expected)
        self.assertIn(annotator_id, range(len(self.annotators)))
    
    def test_cost_tracking(self):
        """Test cumulative cost tracking."""
        samples1 = np.arange(5)
        samples2 = np.arange(5, 10)
        
        result1 = self.manager.annotate_samples(samples1)
        cost1 = self.manager.cumulative_cost

        result2 = self.manager.annotate_samples(samples2)
        cost2 = self.manager.cumulative_cost
        
        self.assertGreater(cost2, cost1)
        self.assertEqual(cost2, cost1 + result2['cost_spent'])


class TestNoisyLabelWrapper(unittest.TestCase):
    """Test NoisyLabelDatasetWrapper."""
    
    def setUp(self):
        self.dataset = MockDataset(size=100, num_classes=10)
        self.annotators = [
            Annotator(self.dataset, "clean", 0.0, 1.0, seed=42),
            Annotator(self.dataset, "sym", 0.3, 0.5, seed=43)
        ]
        self.manager = AnnotatorManager(
            annotators=self.annotators,
            selection_strategy="least_cost",
            seed=42
        )
    
    def test_wrapper_without_annotations(self):
        """Test wrapper returns true labels for unannotated samples."""
        wrapper = NoisyLabelDatasetWrapper(self.dataset, self.manager)
        
        # Sample not yet annotated
        img, label = wrapper[0]
        self.assertEqual(label, self.dataset.targets[0])
    
    def test_wrapper_with_annotations(self):
        """Test wrapper returns noisy labels for annotated samples."""
        # Annotate some samples
        samples = np.array([0, 1, 2, 3, 4])
        self.manager.annotate_samples(samples)
        
        wrapper = NoisyLabelDatasetWrapper(self.dataset, self.manager)
        
        # Check noisy labels are returned
        for idx in samples:
            img, label = wrapper[idx]
            noisy_label = self.manager.get_label_for_sample(idx, self.dataset.targets[idx])
            self.assertEqual(label, noisy_label)
    
    def test_wrapper_preserves_dataset_attributes(self):
        """Test wrapper forwards attributes to wrapped dataset."""
        wrapper = NoisyLabelDatasetWrapper(self.dataset, self.manager)
        
        self.assertEqual(len(wrapper), len(self.dataset))
        self.assertTrue(hasattr(wrapper, 'features'))
        np.testing.assert_array_equal(wrapper.features, self.dataset.features)


class TestMISPAnnotatorIntegration(unittest.TestCase):
    """Test MISP with annotator integration."""
    
    def setUp(self):
        """Setup mock config and data."""
        self.cfg = AttrDict({
            'DATASET': {'NAME': 'CIFAR10'},
            'RNG_SEED': 42,
            'ALPHA': 0.5,
            'DEBUG': False,
            'NORM_IMPORTANCE': False,
            'OWN_ALPHA_WEIGHTING': False,
            'KERNEL_TYPE': 'rbf',
            'SOFT_BORDER_VAL': 0.15
        })
        
        self.dataset = MockDataset(size=100, num_classes=10)
        self.annotators = [
            Annotator(self.dataset, "clean", 0.0, 1.0, seed=42),
            Annotator(self.dataset, "sym", 0.2, 0.5, seed=43),
            Annotator(self.dataset, "sym", 0.3, 0.2, seed=44)
        ]
        self.manager = AnnotatorManager(
            annotators=self.annotators,
            selection_strategy="random",
            seed=42
        )
    
    @patch('deep_al.pycls.datasets.utils.load_features')
    def test_misp_with_annotator_manager(self, mock_load_features):
        """Test MISP accepts annotator_manager parameter."""
        from deep_al.pycls.al.MISP import MISP
        
        # Mock features
        mock_load_features.return_value = torch.randn(100, 128)
        
        misp = MISP(
            cfg=self.cfg,
            budgetSize=10,
            delta=1.0,
            annotator_manager=self.manager
        )
        
        self.assertIsNotNone(misp.annotator_manager)
        self.assertEqual(misp.annotator_manager, self.manager)
    
    @patch('deep_al.pycls.datasets.utils.load_features')
    def test_misp_without_annotator_manager(self, mock_load_features):
        """Test MISP works without annotator_manager (backward compatibility)."""
        from deep_al.pycls.al.MISP import MISP
        
        # Mock features
        mock_load_features.return_value = torch.randn(100, 128)
        
        misp = MISP(
            cfg=self.cfg,
            budgetSize=10,
            delta=1.0,
            annotator_manager=None
        )
        
        self.assertIsNone(misp.annotator_manager)
    
    @patch('deep_al.pycls.datasets.utils.load_features')
    def test_misp_veracity_weighting(self, mock_load_features):
        """Test MISP applies annotator weights to veracity."""
        from deep_al.pycls.al.MISP import MISP
        
        # Mock features
        features = torch.randn(100, 128)
        mock_load_features.return_value = features
        
        # Annotate some samples with different annotators
        lSet = np.array([0, 1, 2, 3, 4])
        self.manager.annotate_with_oracle(lSet[:2])  # First 2 with oracle
        
        # Annotate next 3 with cheap noisy annotator
        for idx in lSet[2:]:
            self.manager.annotator_assignments[int(idx)] = 2  # Noisy annotator
            self.manager.noisy_labels[int(idx)] = self.annotators[2].get_label(idx)
        
        uSet = np.arange(5, 100)
        
        misp = MISP(
            cfg=self.cfg,
            budgetSize=10,
            delta=1.0,
            annotator_manager=self.manager
        )
        
        # Initialize sampling - this should apply annotator weights
        misp.init_sampling_loop(lSet, uSet)
        
        # Check that kernel matrix was weighted (oracle samples should have weight 1.0)
        # This is a smoke test - actual weights are internal
        self.assertTrue(hasattr(misp, 'K'))
        self.assertTrue(hasattr(misp, 'V'))


class TestBackwardCompatibility(unittest.TestCase):
    """Test that non-annotator algorithms still work."""
    
    def setUp(self):
        """Setup mock config."""
        self.cfg = {
            'DATASET': {'NAME': 'CIFAR10'},
            'RNG_SEED': 42,
            'ACTIVE_LEARNING': {
                'BUDGET_SIZE': 10
            }
        }
        self.dataset = MockDataset(size=100, num_classes=10)
    
    @patch('deep_al.pycls.datasets.utils.load_features')
    def test_probcover_no_annotators(self, mock_load_features):
        """Test ProbCover works without annotators."""
        from deep_al.pycls.al.prob_cover import ProbCover
        
        mock_load_features.return_value = np.random.randn(100, 128).astype(np.float32)
        
        probcover = ProbCover(
            cfg=self.cfg,
            budgetSize=10,
            delta=0.6
        )
        
        # Should initialize without errors
        self.assertEqual(probcover.budgetSize, 10)
        self.assertEqual(probcover.delta, 0.6)
    
    @patch('deep_al.pycls.datasets.utils.load_features')
    def test_maxherding_no_annotators(self, mock_load_features):
        """Test MaxHerding works without annotators."""
        from deep_al.pycls.al.maxherding import MaxHerding
        
        mock_load_features.return_value = np.random.randn(100, 128).astype(np.float32)
        
        lSet = np.array([0, 1, 2, 3, 4])
        uSet = np.arange(5, 100)
        
        maxherding = MaxHerding(
            cfg=self.cfg,
            lSet=lSet,
            uSet=uSet,
            budgetSize=10,
            delta=1.0
        )
        
        # Should initialize without errors
        self.assertEqual(maxherding.budgetSize, 10)


class TestLabelPropagation(unittest.TestCase):
    """Test complete label propagation flow."""
    
    def setUp(self):
        self.dataset = MockDataset(size=50, num_classes=10)
        self.annotators = [
            Annotator(self.dataset, "clean", 0.0, 1.0, seed=42),
            Annotator(self.dataset, "sym", 0.3, 0.5, seed=43)
        ]
        self.manager = AnnotatorManager(
            annotators=self.annotators,
            selection_strategy="least_cost",
            seed=42
        )
    
    def test_end_to_end_label_flow(self):
        """Test complete flow: annotation → wrapper → training labels."""
        # 1. Simulate AL selecting samples
        selected_samples = np.array([10, 11, 12, 13, 14])
        
        # 2. Annotate samples (simulates active_sampling_part)
        result = self.manager.annotate_samples(selected_samples)
        
        self.assertEqual(len(result['annotated_samples']), len(selected_samples))
        
        # 3. Wrap dataset
        wrapped_dataset = NoisyLabelDatasetWrapper(self.dataset, self.manager)
        
        # 4. Check training labels are noisy
        for idx in selected_samples:
            img, train_label = wrapped_dataset[idx]
            
            # Get expected noisy label
            annotator_id = self.manager.get_annotator_for_sample(idx)
            expected_label = self.annotators[annotator_id].get_label(idx)
            
            self.assertEqual(train_label, expected_label)
            
            # For noisy annotator, label might differ from true label
            if annotator_id > 0:  # Not oracle
                # Can't guarantee it's different, but we can check it's valid
                self.assertGreaterEqual(train_label, 0)
                self.assertLess(train_label, 10)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAnnotator))
    suite.addTests(loader.loadTestsFromTestCase(TestAnnotatorManager))
    suite.addTests(loader.loadTestsFromTestCase(TestNoisyLabelWrapper))
    suite.addTests(loader.loadTestsFromTestCase(TestMISPAnnotatorIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestBackwardCompatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestLabelPropagation))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
