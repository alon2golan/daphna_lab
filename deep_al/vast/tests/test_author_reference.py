"""Hand-calculated early propagation rounds in the pinned author's code."""
import numpy as np
from scipy import sparse

from deep_al.vast.author_reference import propagate_labels
from deep_al.vast.inference import normalize_evidence, seed_evidence


def test_author_propagates_evidence_without_transmitting_the_uniform_prior():
    kernel = sparse.diags([np.ones(4), np.ones(4)], [-1, 1], shape=(5, 5), format="csr")
    selected, labels = np.array([0]), np.array([0])
    mask = np.arange(5) == 0
    initial = seed_evidence(kernel, selected, labels, 100)
    kwargs = dict(labeled_mask=mask, source_indices=selected, source_labels=labels,
                  alpha=.01, sender_mode="high_confidence", sender_threshold=.01,
                  receiver_mode="high_confidence", receiver_threshold=.75)
    first = propagate_labels(initial, kernel, num_iterations=1, **kwargs)
    # Pure class evidence advances to the two-hop point; other class priors
    # remain .01 rather than being multiplied/transmitted along the edges.
    np.testing.assert_allclose(first[:, 0], [.01, 1.01, 1.01, .01, .01])
    np.testing.assert_allclose(first[:, 1:], .01)
    third = propagate_labels(initial, kernel, num_iterations=3, **kwargs)
    assert normalize_evidence(third)[2, 0] > .75
    np.testing.assert_array_equal(third[mask], initial[mask])


def test_author_uses_known_labels_independently_of_labeled_row_beliefs():
    kernel = sparse.csr_matrix([[0., 1.], [1., 0.]])
    # The labeled row's own belief is deliberately wrong. Its known class,
    # supplied separately, must still seed class 0 at its neighbor.
    initial = np.array([[.01, 4.01], [.01, .01]])
    result = propagate_labels(initial, kernel, np.array([True, False]),
                              np.array([0]), np.array([0]), .01, num_iterations=1)
    np.testing.assert_array_equal(result[0], initial[0])
    np.testing.assert_allclose(result[1], [1.01, .01])
