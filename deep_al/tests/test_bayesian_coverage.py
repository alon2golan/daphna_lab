"""
Unit / correctness tests for BayesianCoverage (design doc §9 layer A).

These exercise the probabilistic core (Steps A/B/C of
docs/bayesian_coverage_DESIGN.md) on tiny synthetic tensors, on CPU, in
seconds. They deliberately bypass BayesianCoverage.__init__ (which loads real
SSL features from disk) by constructing a bare instance and setting only the
handful of attributes the math methods touch. End-to-end pipeline tests
(ProbCover limit, determinism, golden runs — design §9 A6/A7 + B) need the real
feature loader and live separately.

Covered (design §9 A): valid distribution (A1), b_l sanity (A2), single-neighbor
propagation (A3), annotator-accuracy monotonicity (A4), multi-neighbor product
sharpening (A5), numerical safety (A8), plus the coverage_gain acquisition
reducing to a degree ranking at the prior.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from deep_al.pycls.al.bayesian_coverage import BayesianCoverage


def make_bc(num_classes=3, combine='product', neighbor_weight='kernel',
            temper_beta=1.0, selection='coverage_gain'):
    """A bare BayesianCoverage with only the attributes the math methods use."""
    bc = BayesianCoverage.__new__(BayesianCoverage)
    bc.num_of_classes = num_classes
    bc.device = 'cpu'
    bc.matrices_type = torch.float64  # tight tolerances, no fp32 slack
    bc.pi = torch.full((num_classes,), 1.0 / num_classes, dtype=torch.float64)
    bc.log_eps = 1e-12
    bc.combine = combine
    bc.temper_beta = temper_beta
    bc.neighbor_weight = neighbor_weight
    bc.selection = selection
    return bc


def init_C(bc, M):
    """Fresh coverage accumulator (prior), mirroring init_sampling_loop."""
    if bc.combine == 'product':
        return torch.log(bc.pi.clamp_min(bc.log_eps)).repeat(M, 1)
    return bc.pi.unsqueeze(0).repeat(M, 1).clone()


def entropy(p):
    return float(-(p * torch.log(p.clamp_min(1e-12))).sum())


class TestPosteriorValid(unittest.TestCase):
    def test_valid_distribution_both_combines(self):
        # A1: posterior() rows are non-negative and sum to 1 for any C.
        for combine in ('product', 'additive'):
            bc = make_bc(num_classes=4, combine=combine)
            M = 6
            bc.C = init_C(bc, M)
            # perturb with some arbitrary evidence
            k_row = torch.rand(M, dtype=torch.float64)
            bc.C = bc.C + (bc._neighbor_loglik(k_row, 2, 0.7)
                           if combine == 'product' else 0.0)
            if combine == 'additive':
                bc.C[:, 2] += k_row
            P = bc.posterior()
            self.assertEqual(P.shape, (M, 4))
            self.assertTrue(torch.all(P >= 0), f"{combine}: negative prob")
            self.assertTrue(torch.allclose(P.sum(1), torch.ones(M, dtype=torch.float64),
                                           atol=1e-9), f"{combine}: rows !~ 1")


class TestBeliefSanity(unittest.TestCase):
    def test_belief_shape_and_limits(self):
        # A2: b[c]=r, off-diagonal (1-r)/(C-1), sums to 1.
        bc = make_bc(num_classes=5)
        r = 0.6
        b = bc._neighbor_true_label_belief(3, r)
        self.assertAlmostEqual(float(b[3]), r, places=9)
        for k in range(5):
            if k != 3:
                self.assertAlmostEqual(float(b[k]), (1 - r) / 4, places=9)
        self.assertAlmostEqual(float(b.sum()), 1.0, places=9)

    def test_belief_extremes(self):
        bc = make_bc(num_classes=4)
        one_hot = bc._neighbor_true_label_belief(1, 1.0)
        self.assertAlmostEqual(float(one_hot[1]), 1.0, places=9)
        self.assertAlmostEqual(float(one_hot.sum() - one_hot[1]), 0.0, places=9)
        uniform = bc._neighbor_true_label_belief(1, 1.0 / 4)
        self.assertTrue(torch.allclose(uniform, torch.full((4,), 0.25, dtype=torch.float64),
                                       atol=1e-9))


class TestSingleNeighborPropagation(unittest.TestCase):
    def test_propagation_interpolates_in_K(self):
        # A3: with r=1, K=1 -> one-hot on label, K=0 -> prior, monotone between.
        bc = make_bc(num_classes=3, combine='product')
        label = 2
        k_row = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0], dtype=torch.float64)
        M = k_row.numel()
        bc.C = init_C(bc, M) + bc._neighbor_loglik(k_row, label, r=1.0)
        P = bc.posterior()
        # K=1 -> ~one-hot on the label
        self.assertAlmostEqual(float(P[0, label]), 1.0, places=6)
        # K=0 -> prior (uniform)
        self.assertTrue(torch.allclose(P[-1], bc.pi, atol=1e-9))
        # label-prob monotonically non-increasing as K falls 1->0
        label_probs = P[:, label]
        self.assertTrue(torch.all(label_probs[:-1] >= label_probs[1:] - 1e-9),
                        f"not monotone in K: {label_probs}")


class TestAccuracyMonotonicity(unittest.TestCase):
    def test_lower_accuracy_moves_toward_prior(self):
        # A4: fixing geometry, lowering r moves the covered point's posterior
        # toward the prior (peak prob on the label non-increasing in noise).
        bc = make_bc(num_classes=4, combine='product')
        label = 0
        k_row = torch.ones(1, dtype=torch.float64)  # fully covered point
        peaks = []
        for r in [1.0, 0.9, 0.7, 0.5, 1.0 / 4]:
            bc.C = init_C(bc, 1) + bc._neighbor_loglik(k_row, label, r=r)
            peaks.append(float(bc.posterior()[0, label]))
        for a, b in zip(peaks[:-1], peaks[1:]):
            self.assertGreaterEqual(a + 1e-9, b, f"peak rose as accuracy fell: {peaks}")
        # at r=1/C the covered posterior is back at the prior
        self.assertAlmostEqual(peaks[-1], 0.25, places=6)


class TestMultiNeighborSharpening(unittest.TestCase):
    def test_agreeing_neighbors_sharpen(self):
        # A5 (product): two agreeing neighbors -> lower entropy than one alone.
        bc = make_bc(num_classes=3, combine='product')
        label = 1
        r = 0.8
        k = torch.ones(1, dtype=torch.float64)
        one = init_C(bc, 1) + bc._neighbor_loglik(k, label, r)
        two = one + bc._neighbor_loglik(k, label, r)
        bc.C = one
        h1 = entropy(bc.posterior()[0])
        bc.C = two
        h2 = entropy(bc.posterior()[0])
        self.assertLess(h2, h1 - 1e-9, "agreeing neighbors did not sharpen")

    def test_disagreeing_neighbors_raise_entropy(self):
        # A5 (product): two disagreeing neighbors -> higher entropy than one
        # confident neighbor alone.
        bc = make_bc(num_classes=3, combine='product')
        r = 0.8
        k = torch.ones(1, dtype=torch.float64)
        one = init_C(bc, 1) + bc._neighbor_loglik(k, 0, r)
        conflict = one + bc._neighbor_loglik(k, 1, r)
        bc.C = one
        h_one = entropy(bc.posterior()[0])
        bc.C = conflict
        h_conf = entropy(bc.posterior()[0])
        self.assertGreater(h_conf, h_one + 1e-9, "conflict did not raise entropy")

    def test_additive_does_not_sharpen_like_product(self):
        # Documents the property additive lacks: two agreeing votes at K=1 in the
        # additive rule do not drive entropy toward 0 the way product does.
        bc_add = make_bc(num_classes=3, combine='additive')
        bc_add.C = init_C(bc_add, 1)
        k = torch.ones(1, dtype=torch.float64)
        bc_add.C[:, 1] += k
        bc_add.C[:, 1] += k
        h_add = entropy(bc_add.posterior()[0])

        bc_prod = make_bc(num_classes=3, combine='product')
        bc_prod.C = init_C(bc_prod, 1)
        bc_prod.C = bc_prod.C + bc_prod._neighbor_loglik(k, 1, 1.0)
        bc_prod.C = bc_prod.C + bc_prod._neighbor_loglik(k, 1, 1.0)
        h_prod = entropy(bc_prod.posterior()[0])

        self.assertLess(h_prod, h_add, "product should be sharper than additive here")


class TestNumericalSafety(unittest.TestCase):
    def test_no_nan_inf_at_extremes(self):
        # A8: K exactly {0,1} and r near 1 produce no nan/inf.
        bc = make_bc(num_classes=5, combine='product')
        k_row = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.5], dtype=torch.float64)
        bc.C = init_C(bc, k_row.numel())
        bc.C = bc.C + bc._neighbor_loglik(k_row, 3, r=1.0 - 1e-9)
        self.assertTrue(torch.isfinite(bc.C).all(), "non-finite in C")
        P = bc.posterior()
        self.assertTrue(torch.isfinite(P).all(), "non-finite in posterior")
        self.assertTrue(torch.allclose(P.sum(1), torch.ones(5, dtype=torch.float64), atol=1e-9))


class TestAcquisition(unittest.TestCase):
    def test_prior_acquisition_ranks_by_degree(self):
        # At the prior every point is maximally uncertain, so coverage_gain
        # reduces to K @ (constant) i.e. a degree ranking -> argmax = highest
        # out-degree, matching ProbCover's first pick.
        bc = make_bc(num_classes=3, combine='product')
        # 4 points; point 0 connects to everyone, others sparsely.
        K = torch.tensor([
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ], dtype=torch.float64)
        bc.K = K
        bc.C = init_C(bc, 4)
        scores = bc._acquisition_scores()
        self.assertEqual(int(torch.argmax(scores)), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
