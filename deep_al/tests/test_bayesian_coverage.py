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
from deep_al.pycls.noise.annotator import Annotator


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


def loglik(bc, k_row, label, r):
    """Per-point log-likelihood contribution of one symmetric-accuracy-r neighbor."""
    return bc._neighbor_loglik(k_row, bc._likelihood(label, bc._symmetric_confusion(r)))


class TestPosteriorValid(unittest.TestCase):
    def test_valid_distribution_both_combines(self):
        # A1: posterior() rows are non-negative and sum to 1 for any C.
        for combine in ('product', 'additive'):
            bc = make_bc(num_classes=4, combine=combine)
            M = 6
            bc.C = init_C(bc, M)
            # perturb with some arbitrary evidence
            k_row = torch.rand(M, dtype=torch.float64)
            bc.C = bc.C + (loglik(bc, k_row, 2, 0.7)
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
        bc.C = init_C(bc, M) + loglik(bc, k_row, label, 1.0)
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
            bc.C = init_C(bc, 1) + loglik(bc, k_row, label, r)
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
        one = init_C(bc, 1) + loglik(bc, k, label, r)
        two = one + loglik(bc, k, label, r)
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
        one = init_C(bc, 1) + loglik(bc, k, 0, r)
        conflict = one + loglik(bc, k, 1, r)
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
        bc_prod.C = bc_prod.C + loglik(bc_prod, k, 1, 1.0)
        bc_prod.C = bc_prod.C + loglik(bc_prod, k, 1, 1.0)
        h_prod = entropy(bc_prod.posterior()[0])

        self.assertLess(h_prod, h_add, "product should be sharper than additive here")


class TestNumericalSafety(unittest.TestCase):
    def test_no_nan_inf_at_extremes(self):
        # A8: K exactly {0,1} and r near 1 produce no nan/inf.
        bc = make_bc(num_classes=5, combine='product')
        k_row = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.5], dtype=torch.float64)
        bc.C = init_C(bc, k_row.numel())
        bc.C = bc.C + loglik(bc, k_row, 3, 1.0 - 1e-9)
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


class _ToyDataset:
    """Minimal dataset with a `targets` attribute for Annotator construction."""
    def __init__(self, targets):
        self.targets = list(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return 0, self.targets[i]


class TestAnnotatorConfusionMatrix(unittest.TestCase):
    def test_clean_is_identity(self):
        ann = Annotator(_ToyDataset([0, 1, 2, 0, 1, 2]), noise_type='clean',
                        noise_rate=0.0, num_classes=3)
        self.assertTrue(np.allclose(ann.confusion_matrix, np.eye(3)))

    def test_symmetric_matches_closed_form(self):
        C, nr = 4, 0.3
        ann = Annotator(_ToyDataset(list(range(C)) * 5), noise_type='sym',
                        noise_rate=nr, num_classes=C, seed=1)
        M = ann.confusion_matrix
        self.assertTrue(np.allclose(np.diag(M), 1.0 - nr))       # diag = accuracy
        off = M[~np.eye(C, dtype=bool)]
        self.assertTrue(np.allclose(off, nr / (C - 1)))          # uniform off-diag
        self.assertTrue(np.allclose(M.sum(1), 1.0))

    def test_asymmetric_uses_transition_matrix(self):
        # Circular flip k -> k+1 for the noised fraction.
        C, nr = 3, 0.4
        T = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        ann = Annotator(_ToyDataset(list(range(C)) * 5), noise_type='asym',
                        noise_rate=nr, num_classes=C, seed=1,
                        noise_transition_matrix=T)
        M = ann.confusion_matrix
        expected = (1 - nr) * np.eye(C) + nr * T
        self.assertTrue(np.allclose(M, expected))
        self.assertTrue(np.allclose(M.sum(1), 1.0))
        # off-diagonal mass sits only on the flip target, unlike symmetric noise.
        self.assertAlmostEqual(M[0, 1], nr)
        self.assertAlmostEqual(M[0, 2], 0.0)


class TestBeliefFromConfusion(unittest.TestCase):
    def test_belief_matches_bayes_rule(self):
        # b(k) ∝ pi(k) * M[k, c]; check against a hand-computed asym matrix.
        bc = make_bc(num_classes=3)
        M = torch.tensor([[0.7, 0.3, 0.0],
                          [0.0, 0.8, 0.2],
                          [0.1, 0.0, 0.9]], dtype=torch.float64)
        c = 1  # observed label
        b = bc._belief(c, M)
        # uniform prior -> b ∝ column c of M = [0.3, 0.8, 0.0]
        expected = torch.tensor([0.3, 0.8, 0.0], dtype=torch.float64)
        expected = expected / expected.sum()
        self.assertTrue(torch.allclose(b, expected, atol=1e-9))
        self.assertAlmostEqual(float(b.sum()), 1.0, places=9)

    def test_symmetric_confusion_reduces_to_scalar_form(self):
        # _confusion_matrix built from a real sym annotator -> _belief(c) peak = accuracy.
        C, nr = 4, 0.25
        ann = Annotator(_ToyDataset(list(range(C)) * 5), noise_type='sym',
                        noise_rate=nr, num_classes=C, seed=2)

        class _MgrStub:
            annotators = [ann]
            annotator_accuracies = [ann.accuracy]
        bc = make_bc(num_classes=C)
        bc.annotator_manager = _MgrStub()
        M = bc._confusion_matrix(0)
        b = bc._belief(2, M)
        self.assertAlmostEqual(float(b[2]), ann.accuracy, places=9)


class _StubAnn:
    def __init__(self, M): self._M = M
    @property
    def confusion_matrix(self): return self._M


class _StubMgr:
    def __init__(self, M, labels):
        self.annotators = [_StubAnn(M)]
        self.annotator_accuracies = [float(np.diag(M)[0])]
        self.annotated_labels = labels          # {global_idx: observed_label}
    def get_annotator_for_sample(self, g): return 0
    def select_annotator(self): return 0


def _bc_with_posterior():
    """
    A product BayesianCoverage whose persisted accumulator C_general encodes,
    for 3 labeled points, a self-vote (from the observed label at accuracy 0.8)
    plus a neighbour boost:
      pt0 observed=0, neighbours boost 0  -> LOO agrees   (clean)
      pt1 observed=0, neighbours boost 1  -> LOO disagrees (noisy; map flips->1)
      pt2 observed=1, neighbours boost 1  -> LOO agrees   (clean)
    """
    C = 3
    r = 0.8
    M = np.full((C, C), (1 - r) / (C - 1)); np.fill_diagonal(M, r)
    bc = make_bc(num_classes=C, combine='product')
    bc.annotator_manager = _StubMgr(M, {0: 0, 1: 0, 2: 1})
    bc.train_label_mode = 'observed'
    logpi = torch.log(bc.pi)
    def belief(c):
        b = torch.full((C,), (1 - r) / (C - 1), dtype=torch.float64); b[c] = r
        return torch.log(b)
    rows = torch.stack([
        logpi + belief(0) + torch.tensor([2., 0., 0.], dtype=torch.float64),
        logpi + belief(0) + torch.tensor([0., 3., 0.], dtype=torch.float64),
        logpi + belief(1) + torch.tensor([0., 2., 0.], dtype=torch.float64),
    ])
    # C_general is indexed by GLOBAL idx; points live at global 0,1,2.
    bc.C_general = rows
    return bc


class TestTrainingTargets(unittest.TestCase):
    def test_observed_passthrough(self):
        bc = _bc_with_posterior()
        self.assertEqual(bc.training_targets([0, 1, 2], 'observed'), {'mode': 'observed'})

    def test_map_posterior_relabels_on_neighbour_override(self):
        bc = _bc_with_posterior()
        out = bc.training_targets([0, 1, 2], 'map_posterior')
        # pt1's strong disagreeing neighbours flip its label 0 -> 1; others stay.
        self.assertEqual(out['hard_labels'], {0: 0, 1: 1, 2: 1})

    def test_filtered_drops_neighbour_disagreements(self):
        bc = _bc_with_posterior()
        out = bc.training_targets([0, 1, 2], 'filtered')
        # LOO: pt0 (obs0, boost0) and pt2 (obs1, boost1) agree -> kept; pt1 dropped.
        self.assertEqual(sorted(out['keep']), [0, 2])

    def test_soft_posterior_returns_valid_distributions(self):
        bc = _bc_with_posterior()
        out = bc.training_targets([0, 1, 2], 'soft_posterior')
        for g in (0, 1, 2):
            v = out['soft'][g]
            self.assertEqual(v.shape, (3,))
            self.assertAlmostEqual(float(v.sum()), 1.0, places=6)
        # pt0 soft target peaks on class 0
        self.assertEqual(int(out['soft'][0].argmax()), 0)

    def test_reweighted_weights_by_loo_reliability(self):
        bc = _bc_with_posterior()
        out = bc.training_targets([0, 1, 2], 'reweighted')
        # each soft target is w*onehot(observed): mass on the observed class only
        self.assertEqual(int(out['soft'][0].argmax()), 0)   # observed 0
        self.assertEqual(int(out['soft'][2].argmax()), 1)   # observed 1
        w0 = out['soft'][0][0]      # pt0 neighbours agree -> high reliability
        w1 = out['soft'][1][0]      # pt1 neighbours disagree -> low reliability
        self.assertGreater(w0, w1)
        self.assertTrue(0.0 <= w1 <= w0 <= 1.0)


class _FakeAM:
    """Minimal annotator-manager stub: symmetric accuracy via the fallback path."""
    def __init__(self, annotated_labels, acc=0.9):
        self.annotated_labels = annotated_labels
        self.annotators = {0: object()}          # no .confusion_matrix -> fallback
        self.annotator_accuracies = {0: acc}

    def get_annotator_for_sample(self, g):
        return 0


class TestNWOutOfBall(unittest.TestCase):
    """§7.9 out-of-ball propagation (noise-aware soft-kNN)."""

    def _bc(self, acc=0.9):
        bc = make_bc(num_classes=3)
        bc.out_of_ball = 'nw'
        bc.nw_m = 7
        bc.nw_kappa = 3.0
        bc.oob_weight_mode = 'const'
        bc.oob_weight = 1.0
        # 1-D feature line: anchors at 0.0 (label 0), 10.0 (label 1),
        # 30.0 (label 2); uncovered points at 2.0 (near anchor A), 100.0
        # (far from EVERYTHING, relatively nearest to anchor C), and 5.0
        # (exactly equidistant between anchors A and B).
        bc.all_features = np.array(
            [[0.0], [10.0], [30.0], [2.0], [100.0], [5.0]], dtype=np.float64)
        bc.annotator_manager = _FakeAM({0: 0, 1: 1, 2: 2}, acc=acc)
        return bc

    def test_valid_distributions_and_commitment(self):
        bc = self._bc()
        out = bc._nw_targets([3, 4, 5])
        self.assertEqual(set(out), {3, 4, 5})
        for P in out.values():
            self.assertAlmostEqual(float(P.sum()), 1.0, places=6)
            self.assertTrue((P >= 0).all())
        # near point follows anchor A's belief (class 0)
        self.assertEqual(int(out[3].argmax()), 0)

    def test_influence_never_vanishes(self):
        # the point at 100.0 is far outside ANY plausible delta of its nearest
        # anchor (30.0), yet still commits to it — gap-scaled competition
        # means only RELATIVE distances matter (70 vs 90 vs 100).
        bc = self._bc()
        P_far = bc._nw_targets([4])[4]
        self.assertEqual(int(P_far.argmax()), 2)
        self.assertGreater(float(P_far[2]), 0.5)

    def test_equidistant_anchors_share_weight(self):
        # the point at 5.0 sits exactly between anchors A (label 0) and B
        # (label 1): the two nearest weights tie, so neither class can carry
        # more than ~half the belief mass.
        bc = self._bc()
        P = bc._nw_targets([5])[5]
        self.assertAlmostEqual(float(P[0]), float(P[1]), places=6)
        self.assertLess(float(P.max()), 0.55)

    def test_mixture_bounded_by_belief_sharpness(self):
        # NW is a mixture: it can never be sharper than the sharpest belief.
        bc = self._bc(acc=0.6)
        P = bc._nw_targets([3])[3]
        self.assertLessEqual(float(P.max()), 0.6 + 1e-9)

    def test_oob_weight_scales_row_sum(self):
        # OOB_WEIGHT scales the soft-target row: soft-CE applies the row sum as
        # the per-sample loss weight, so far targets get 0.25x the gradient.
        bc = self._bc()
        bc.oob_weight = 0.25
        out = bc._nw_targets([3, 4])
        for P in out.values():
            self.assertAlmostEqual(float(P.sum()), 0.25, places=6)
        # relative class distribution (argmax) unchanged by the scaling
        self.assertEqual(int(out[3].argmax()), 0)

    def test_topk_scope_raises(self):
        bc = self._bc()
        bc.pseudo_scope = 'topk'
        bc.relevant_indices = np.arange(6)
        bc.C_general = torch.zeros(6, 3, dtype=torch.float64)  # all prior-only
        with self.assertRaises(NotImplementedError):
            bc.pseudo_label_targets([])

    def test_all_scope_all_uncovered_returns_nw(self):
        bc = self._bc()
        bc.pseudo_scope = 'all'
        bc.relevant_indices = np.arange(6)
        bc.C_general = torch.zeros(6, 3, dtype=torch.float64)  # all prior-only
        out = bc.pseudo_label_targets([])
        # anchors (annotated 0,1,2) excluded; uncovered 3,4,5 get NW targets
        self.assertEqual(set(out), {3, 4, 5})
        self.assertEqual(int(out[3].argmax()), 0)
        self.assertEqual(int(out[4].argmax()), 2)


class TestNonUniformPriorNoCollapse(unittest.TestCase):
    """
    Regression for the 2026-07-29 PoE prior double-count bug: pi must enter
    the product exactly once (via the C init), never per neighbor. The old
    code used the Step-A posterior (pi inside) as each expert AND pi as the
    out-of-ball neutral element, so N annotations multiplied pi^(N+1) and the
    posterior collapsed to argmax pi under any non-uniform prior.
    """

    def _bc_with_prior(self, pi, **kw):
        bc = make_bc(num_classes=len(pi), combine='product', **kw)
        bc.pi = torch.tensor(pi, dtype=torch.float64)
        return bc

    def test_far_neighbors_are_evidence_free(self):
        # 50 out-of-ball (K=0) annotations must leave the posterior AT the
        # prior — under the old code it sharpened to argmax pi. equal
        # weighting (w=1): kernel weighting already zeroes K=0 contributions,
        # so it wouldn't exercise the neutral element.
        bc = self._bc_with_prior([0.7, 0.2, 0.1], neighbor_weight='equal')
        bc.C = init_C(bc, 1)
        k_row = torch.zeros(1, dtype=torch.float64)
        for _ in range(50):
            bc.C = bc.C + loglik(bc, k_row, 1, 0.9)
        self.assertTrue(torch.allclose(bc.posterior()[0], bc.pi, atol=1e-9))

    def test_single_inball_label_is_one_bayes_update(self):
        # One K=1 neighbor -> posterior = pi * M[:, label] / Z: the prior
        # applied exactly once, not squared.
        bc = self._bc_with_prior([0.7, 0.2, 0.1])
        r, label = 0.9, 1
        bc.C = init_C(bc, 1) + loglik(bc, torch.ones(1, dtype=torch.float64), label, r)
        M = bc._symmetric_confusion(r)
        expected = bc.pi * M[:, label]
        expected = expected / expected.sum()
        self.assertTrue(torch.allclose(bc.posterior()[0], expected, atol=1e-9))

    def test_uniform_prior_unchanged_by_fix(self):
        # Under uniform pi the likelihood expert equals the old posterior
        # expert and 1/C equals the old pi neutral term, so the fix is a
        # no-op: q(u,k) = K*b(k) + (1-K)*pi(k) reproduced exactly.
        bc = make_bc(num_classes=4, combine='product')
        k_row = torch.tensor([1.0, 0.6, 0.3, 0.0], dtype=torch.float64)
        r, label = 0.8, 2
        new = loglik(bc, k_row, label, r)
        b_old = bc._neighbor_true_label_belief(label, r)      # pi * M / Z
        q_old = (k_row.unsqueeze(1) * b_old.unsqueeze(0)
                 + (1.0 - k_row.unsqueeze(1)) * bc.pi.unsqueeze(0))
        old = bc.temper_beta * k_row.unsqueeze(1) * torch.log(q_old.clamp_min(bc.log_eps))
        self.assertTrue(torch.allclose(new, old, atol=1e-12))


if __name__ == '__main__':
    unittest.main(verbosity=2)
