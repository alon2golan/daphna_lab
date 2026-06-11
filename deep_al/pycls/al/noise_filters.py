"""
Pluggable noise filters for Noise-Aware Sampling (NPC) and train-on-clean.

A noise filter looks at the current labeled set (whose labels may be noisy, as
served by the annotator system) and predicts, per labeled point, whether its
label is clean. This is used in two places (Netta's thesis):

  1. Query side (NPC): between inner batches, noisy-predicted points are dropped
     from the coverage so their region re-opens for selection.
  2. Training side (train-on-clean): noisy-predicted points are dropped before
     the supervised training step.

All filters share one interface:

    predict_clean(cfg, train_data, l_set, u_set, annotator_manager) -> np.ndarray[bool]

returning a boolean mask aligned with `l_set` (True == predicted clean).
"""
import numpy as np

from deep_al.pycls.utils.AUM import AUM


class NoiseFilter:
    """Base class: predict which labeled points carry a clean label."""

    def predict_clean(self, cfg, train_data, l_set, u_set, annotator_manager):
        raise NotImplementedError


class NoOpNoiseFilter(NoiseFilter):
    """Treats every labeled point as clean. Equivalent to disabling filtering."""

    def predict_clean(self, cfg, train_data, l_set, u_set, annotator_manager):
        return np.ones(len(l_set), dtype=bool)


class IdealNoiseFilter(NoiseFilter):
    """
    Oracle filter (upper bound): a labeled point is clean iff the label the
    annotator actually assigned equals the ground-truth label. Uses the true
    labels, so it is only valid as a baseline to confirm NPC actually helps.
    """

    def predict_clean(self, cfg, train_data, l_set, u_set, annotator_manager):
        true_labels = _get_true_labels(train_data, annotator_manager)
        assigned = annotator_manager.annotated_labels
        clean = np.array(
            [int(assigned[int(g)]) == int(true_labels[int(g)]) for g in l_set],
            dtype=bool,
        )
        return clean


class LowBudgetAUMFilter(NoiseFilter):
    """
    LowBudgetAUM (Netta's thesis): train a linear classifier on SSL features with
    an extra fake class; labeled points whose Area-Under-the-Margin falls below a
    fake-class-calibrated threshold are flagged noisy. Wraps the existing AUM.py.
    """

    def predict_clean(self, cfg, train_data, l_set, u_set, annotator_manager):
        nf = cfg.NOISE_FILTER
        aum = AUM(
            train_data=train_data,
            l_set=np.asarray(l_set, dtype=int),
            u_set=np.asarray(u_set, dtype=int),
            cfg=cfg,
            num_repeats=nf.AUM_NUM_REPEATS,
            thr_percentile=nf.AUM_THR_PERCENTILE,
            epochs=nf.AUM_EPOCHS,
            batch_size=nf.AUM_BATCH_SIZE,
            linear_from_features=True,
        )
        # AUM.l_set_is_clean is aligned with the l_set order passed in.
        return np.asarray(aum.l_set_is_clean, dtype=bool)


def _get_true_labels(train_data, annotator_manager):
    """Ground-truth labels indexed by global sample index."""
    if annotator_manager is not None and getattr(annotator_manager, "annotators", None):
        # All annotators wrap the same dataset, so clean_labels is identical.
        return np.asarray(annotator_manager.annotators[0].clean_labels)
    return np.asarray(train_data.targets)


def build_noise_filter(cfg):
    """Factory: pick a noise filter from cfg.NOISE_FILTER.METHOD."""
    method = str(cfg.NOISE_FILTER.METHOD).lower()
    if method in ("aum", "lowbudgetaum", "low_budget_aum"):
        return LowBudgetAUMFilter()
    if method == "ideal":
        return IdealNoiseFilter()
    if method in ("none", "noop", ""):
        return NoOpNoiseFilter()
    raise NotImplementedError(f"Unknown NOISE_FILTER.METHOD: {cfg.NOISE_FILTER.METHOD}")
