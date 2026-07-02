"""
Bayesian Coverage for noisy-annotator active learning.

A probabilistic generalization of `CoverageMatrixMethod` (see
`coverage_matrix_methods.py`). Full specification and the mathematical
derivation live in `daphna_lab/docs/bayesian_coverage_DESIGN.md`.

Summary of the model (design doc §2):
  * Each point carries a per-class posterior over its label instead of a
    binary "covered / not covered" flag.
  * A labeled neighbor `l` (observed label `y~_l` from annotator `a`) yields a
    posterior over its OWN true label b_l(k) that depends on the annotator's
    accuracy (Step A). Near-oracle annotator -> peaked; noisy annotator -> flat.
  * That belief propagates to a point `u` through the kernel edge K[u,l]:
        q_l(u,k) = K[u,l]*b_l(k) + (1-K[u,l])*pi(k)          (Step B)
  * All labeled neighbors are fused as a PRODUCT OF EXPERTS in log-space
    (Step C). This is the mathematically-correct combination but it REPLACES
    Itai's additive rule -- see design doc §3: the `additive` branch is kept and
    MUST be compared empirically before `product` is adopted as default.

STATUS: implemented for design build-order steps 1-2. Working: `product` /
`additive` combine, `coverage_gain` selection, `prior` reliability, and the full
confusion-matrix Step A (symmetric AND asymmetric annotators, via
`Annotator.confusion_matrix`). Marked TODO / NotImplementedError:
  * entropy (expected-info) selection            (design §6)
  * TRAIN_LABEL_MODE target construction          (design §5)
  * sparse-K path
  * NoiseAwareBayesianCoverage posterior reliability r_l  (design §4, FUTURE)
"""
import numpy as np
import torch
from tqdm import tqdm

from deep_al.pycls.al.coverage_matrix_methods import CoverageMatrixMethod, NoiseAwareCoverage


class BayesianCoverage(CoverageMatrixMethod):
    """
    Soft, per-class Bayesian coverage (design doc points 1-3).

    Reuses the base class for kernel construction (`K_general`), index
    bookkeeping (`set_rel_features`, `init_sampling_loop`) and finalization
    (`_finalize_selection`); overrides the coverage representation, its update
    rule, and the acquisition score.

    The coverage accumulator `self.C` (shape [M, num_classes], M = |relevant|)
    is interpreted according to `COMBINE`:
        * "product":  log-unnormalized class posterior; init = log(pi);
                      update += w * log q_l;  posterior = softmax(C).
        * "additive": linear pseudo-counts (Itai's rule); init = alpha;
                      update: C[:, label] += K_row;  posterior = C / C.sum(1).
    """

    def __init__(self, cfg, annotator_manager=None):
        super().__init__(cfg, annotator_manager=annotator_manager)

        bc = getattr(cfg, 'BAYES_COVERAGE', None)

        def _get(key, default):
            return getattr(bc, key, default) if bc is not None else default

        # --- config (design doc §7) ---
        self.combine = _get('COMBINE', 'product')            # product | additive
        self.prior_kind = _get('PRIOR', 'uniform')           # uniform | empirical
        self.temper_beta = float(_get('TEMPER_BETA', 1.0))   # product tempering
        self.neighbor_weight = _get('NEIGHBOR_WEIGHT', 'kernel')  # kernel | equal
        self.selection = _get('SELECTION', 'coverage_gain')  # coverage_gain | entropy
        self.reliability_source = _get('RELIABILITY_SOURCE', 'prior')  # prior|graph|aum
        self.train_label_mode = _get('TRAIN_LABEL_MODE', 'observed')

        if self.use_sparse:
            raise NotImplementedError(
                "BayesianCoverage: sparse-K path not implemented yet. "
                "Set SPARSE_K=False."
            )
        if self.reliability_source != 'prior':
            raise NotImplementedError(
                f"RELIABILITY_SOURCE={self.reliability_source!r} is future work "
                "(design §4). Use 'prior' for now."
            )

        C = self.num_of_classes
        if self.prior_kind == 'uniform':
            pi = torch.full((C,), 1.0 / C, device=self.device, dtype=self.matrices_type)
        else:
            raise NotImplementedError("PRIOR='empirical' not implemented yet.")
        self.pi = pi
        self.log_eps = 1e-12
        # Becomes True once a selection has populated C_general, so training-time
        # posterior queries (training_targets) don't run on an all-prior C_general.
        self._selection_done = False

    # ------------------------------------------------------------------ #
    # coverage representation
    # ------------------------------------------------------------------ #
    def init_sampling_loop(self, lset, uset):
        # Reuse base to build self.K and index bookkeeping, then reinterpret C.
        super().init_sampling_loop(lset, uset)
        M = len(self.relevant_indices)
        if self.combine == 'product':
            # log-prior broadcast over all points
            self.C = torch.log(self.pi.clamp_min(self.log_eps)).repeat(M, 1)
        elif self.combine == 'additive':
            # Dirichlet-style prior pseudocounts = pi (rows already sum to 1, so
            # the initial posterior is the prior and posterior() is always a
            # valid distribution — unlike an all-zero init).
            self.C = self.pi.unsqueeze(0).repeat(M, 1).clone()
        else:
            raise ValueError(f"Unknown COMBINE={self.combine!r}")

        # Base CoverageMatrixMethod carries coverage across AL episodes via
        # C_general; we reset C to the prior each episode, so we must instead
        # re-fold every already-labeled point (the current lSet) or the posterior
        # would forget everything selected in previous episodes.
        self._fold_labeled_set()

    def posterior(self):
        """Per-point class posterior P(y_u = .) as a [M, num_classes] tensor."""
        if self.combine == 'product':
            return torch.softmax(self.C, dim=1)
        # additive
        return self.C / self.C.sum(dim=1, keepdim=True).clamp_min(self.log_eps)

    def _fold_labeled_set(self):
        """
        Re-accumulate the evidence of every already-labeled point (the first
        len(self.lSet) entries of relevant_indices) into the freshly-initialized
        C, using each point's recorded observed label and annotator. Points with
        no recorded annotation (e.g. an unannotated oracle warm-start) are
        skipped — there is no observed label to fold.
        """
        am = self.annotator_manager
        if am is None:
            return
        for local in range(len(self.lSet)):
            global_idx = int(self.relevant_indices[local])
            if global_idx not in am.annotated_labels:
                continue
            chosen_label = int(am.annotated_labels[global_idx])
            annotator_id = am.get_annotator_for_sample(global_idx)
            self._fold_neighbor(local, chosen_label, annotator_id)

    # ------------------------------------------------------------------ #
    # per-neighbor evidence (design §2 Steps A + B)
    # ------------------------------------------------------------------ #
    def _confusion_matrix(self, annotator_id):
        """
        M[k, j] = P(observed = j | true = k) for the given annotator, as a
        [C, C] tensor. Uses the annotator's own effective confusion matrix
        (`Annotator.confusion_matrix`, which handles clean / sym / asym), and
        falls back to a symmetric matrix built from the recorded accuracy for
        annotator objects that don't expose it.
        """
        if annotator_id is None:
            annotator_id = self.annotator_manager.select_annotator()
        annotator = self.annotator_manager.annotators[annotator_id]
        M_np = getattr(annotator, 'confusion_matrix', None)
        if M_np is None:
            acc = float(self.annotator_manager.annotator_accuracies[annotator_id])
            return self._symmetric_confusion(acc)
        return torch.as_tensor(M_np, device=self.device, dtype=self.matrices_type)

    def _symmetric_confusion(self, r):
        """Symmetric confusion matrix with accuracy r on the diagonal and
        (1-r)/(C-1) off-diagonal — the closed form of a `sym` annotator."""
        C = self.num_of_classes
        M = torch.full((C, C), (1.0 - r) / (C - 1), device=self.device, dtype=self.matrices_type)
        M.fill_diagonal_(r)
        return M

    def _belief(self, chosen_label, M):
        """
        b_l(k) = P(y_l = k | y~_l = chosen_label)  ∝  pi(k) * M[k, chosen_label]
        (design §2 Step A). General form: annotator accuracy enters through the
        confusion matrix M, so symmetric and asymmetric annotators are handled
        uniformly. Returns a normalized [C] tensor.
        """
        b = self.pi * M[:, chosen_label]                                # [C]
        return b / b.sum().clamp_min(self.log_eps)

    def _neighbor_true_label_belief(self, chosen_label, r):
        """
        Convenience wrapper: Step-A belief for a symmetric annotator of accuracy
        r. Derived from the general `_belief` path so the symmetric closed form
        (b(chosen_label) = r) is validated against the confusion-matrix form.
        """
        return self._belief(chosen_label, self._symmetric_confusion(r))

    def _neighbor_loglik(self, k_row, b):
        """
        Weighted log-likelihood contribution of one labeled neighbor to every
        point (design §2 Steps B + C, §2.4 weighting), given its true-label
        belief b (=[C], from `_belief`):
            q(u,k) = K[u,l]*b(k) + (1-K[u,l])*pi(k)
            return w_u * log q(u,k)                       shape [M, num_classes]
        """
        k = k_row.unsqueeze(1)                                          # [M,1]
        q = k * b.unsqueeze(0) + (1.0 - k) * self.pi.unsqueeze(0)       # [M,C]
        logq = torch.log(q.clamp_min(self.log_eps))
        if self.neighbor_weight == 'kernel':
            w = k_row.unsqueeze(1)
        elif self.neighbor_weight == 'equal':
            w = 1.0
        else:
            raise ValueError(f"Unknown NEIGHBOR_WEIGHT={self.neighbor_weight!r}")
        return self.temper_beta * w * logq

    # ------------------------------------------------------------------ #
    # acquisition (design §6)
    # ------------------------------------------------------------------ #
    def _acquisition_scores(self):
        """
        coverage_gain: prioritize candidates that sit near still-uncertain mass,
            score[v] = sum_u K[v,u] * uncertainty(u),  uncertainty = 1 - max_k P.
        Returns a [M] tensor of scores (higher = better).
        """
        if self.selection == 'coverage_gain':
            P = self.posterior()
            uncertainty = 1.0 - P.max(dim=1).values                    # [M]
            return self.K @ uncertainty                                # [M]
        if self.selection == 'entropy':
            raise NotImplementedError(
                "SELECTION='entropy' (expected uncertainty reduction) is a "
                "planned second variant (design §6)."
            )
        raise ValueError(f"Unknown SELECTION={self.selection!r}")

    # ------------------------------------------------------------------ #
    # fold one labeled point into the coverage accumulator
    # ------------------------------------------------------------------ #
    def _fold_neighbor(self, local_idx, chosen_label, annotator_id):
        """
        Fold the evidence of one labeled point into C. `local_idx` indexes into
        relevant_indices, `chosen_label` is its observed (possibly noisy) label,
        `annotator_id` identifies the annotator (its confusion matrix supplies
        the accuracy for Step A). Shared by the in-loop greedy step and the
        episode-start re-fold of the labeled set.
        """
        k_row = self.K[local_idx].to_dense().squeeze()
        if self.combine == 'product':
            b = self._belief(chosen_label, self._confusion_matrix(annotator_id))
            self.C += self._neighbor_loglik(k_row, b)
        else:  # additive (Itai's rule, clean-label baseline)
            self.C[:, chosen_label] += k_row
        del k_row

    # ------------------------------------------------------------------ #
    # greedy step
    # ------------------------------------------------------------------ #
    def _select_one(self, selected, annotator_id):
        curr_l_set = np.concatenate((np.arange(len(self.lSet)), selected)).astype(int)

        scores = self._acquisition_scores()
        scores[curr_l_set] = -np.inf
        sampled_point = torch.argmax(scores).item()

        global_idx = int(self.relevant_indices[sampled_point])
        result = self.annotator_manager.annotate_samples([global_idx], annotator_id)
        chosen_label = result['annotated_labels'][global_idx]
        # annotate_samples resolves None -> a concrete annotator; record which one
        # so the confusion matrix matches the annotator that actually labelled it.
        used_annotator = result['annotator_assignments'][global_idx]

        self._fold_neighbor(sampled_point, chosen_label, used_annotator)

        assert sampled_point not in selected, 'sample was already selected'
        selected.append(sampled_point)
        return sampled_point, global_idx, chosen_label

    # ------------------------------------------------------------------ #
    # per-point posterior over GLOBAL indices (read after selection)
    # ------------------------------------------------------------------ #
    # select_samples persists the final per-point accumulator into
    # self.C_general[self.relevant_indices] before deleting self.C, so the
    # posterior for any point survives the episode and can be queried at
    # training time (train_al calls training_targets after selection).

    def _finalize_selection(self, selected, num_samples):
        # Mark that C_general now holds real posteriors (base method untouched).
        self._selection_done = True
        return super()._finalize_selection(selected, num_samples)

    def _accum_rows(self, global_indices):
        idx = torch.as_tensor([int(g) for g in global_indices],
                              device=self.device, dtype=torch.long)
        return self.C_general[idx]                                # [k, C]

    def posterior_for(self, global_indices):
        """Per-point posterior P(y=·) for the given GLOBAL indices -> [k, C]."""
        rows = self._accum_rows(global_indices)
        if self.combine == 'product':
            return torch.softmax(rows, dim=1)
        return rows / rows.sum(dim=1, keepdim=True).clamp_min(self.log_eps)

    def _loo_posterior_for(self, global_indices):
        """
        Leave-one-out posterior: the same posterior with each point's OWN
        observed-label vote removed, so it reflects only that point's NEIGHBORS
        (the graph-based reliability signal, design §4b). This is what makes a
        labeled point's cleanliness judgeable — the full posterior is dominated
        by the point's own (self-loop K=1) vote and would rarely disagree.
        """
        rows = self._accum_rows(global_indices).clone()
        am = self.annotator_manager
        for i, g in enumerate(global_indices):
            g = int(g)
            if am is None or g not in am.annotated_labels:
                continue
            c = int(am.annotated_labels[g])
            if self.combine == 'product':
                # self-term = temper_beta * w_ll * log q_ll; the self-loop has
                # K[l,l]=1 (distance 0, tophat or rbf) so q_ll=b_l and w_ll=1.
                b = self._belief(c, self._confusion_matrix(am.get_annotator_for_sample(g)))
                rows[i] -= self.temper_beta * torch.log(b.clamp_min(self.log_eps))
            else:
                rows[i, c] -= 1.0                                 # additive: drop one count
        if self.combine == 'product':
            return torch.softmax(rows, dim=1)
        rows = rows.clamp_min(0.0)
        return rows / rows.sum(dim=1, keepdim=True).clamp_min(self.log_eps)

    # ------------------------------------------------------------------ #
    # training-target policy (design §5) -- CONFIGURABLE, to be compared
    # ------------------------------------------------------------------ #
    def training_targets(self, global_indices, mode=None):
        """
        Build training targets for the given labeled GLOBAL indices according to
        TRAIN_LABEL_MODE (design §5) -- an experimental arm to compare, not a
        baked-in choice. Returns a dict describing how train_al should build the
        training loader:

          observed       -> {'mode':'observed'}                      (raw noisy labels)
          map_posterior  -> {'mode', 'hard_labels': {gidx: int}}     (argmax posterior)
          filtered       -> {'mode', 'keep': [gidx, ...]}            (LOO-clean subset)
          soft_posterior -> {'mode', 'soft': {gidx: np[C]}}          (posterior as soft label)
          reweighted     -> {'mode', 'soft': {gidx: np[C]}}          (w*onehot(observed),
                                                                      w = LOO reliability)

        The clean/reliability signal is the method's OWN posterior (design §4b,
        no AUM); reweighted/filtered use the leave-one-out posterior so a point's
        own vote does not mask a disagreeing neighbourhood.
        """
        mode = mode or self.train_label_mode
        am = self.annotator_manager
        gidxs = [int(g) for g in global_indices]
        observed = {g: int(am.annotated_labels[g]) for g in gidxs
                    if am is not None and g in am.annotated_labels}

        if mode == 'observed':
            return {'mode': 'observed'}

        if mode == 'map_posterior':
            P = self.posterior_for(gidxs)
            hard = {g: int(P[i].argmax()) for i, g in enumerate(gidxs)}
            return {'mode': mode, 'hard_labels': hard}

        if mode == 'filtered':
            Ploo = self._loo_posterior_for(gidxs)
            keep = [g for i, g in enumerate(gidxs)
                    if g in observed and int(Ploo[i].argmax()) == observed[g]]
            return {'mode': mode, 'keep': keep}

        if mode == 'soft_posterior':
            P = self.posterior_for(gidxs)
            soft = {g: P[i].detach().cpu().numpy() for i, g in enumerate(gidxs)}
            return {'mode': mode, 'soft': soft}

        if mode == 'reweighted':
            Ploo = self._loo_posterior_for(gidxs)
            soft = {}
            for i, g in enumerate(gidxs):
                if g not in observed:
                    continue
                w = float(Ploo[i, observed[g]])          # reliability of the observed label
                v = np.zeros(self.num_of_classes, dtype=np.float32)
                v[observed[g]] = w
                soft[g] = v
            return {'mode': mode, 'soft': soft}

        raise ValueError(
            f"Unknown TRAIN_LABEL_MODE={mode!r}. Expected one of: observed, "
            "map_posterior, filtered, soft_posterior, reweighted.")


class NoiseAwareBayesianCoverage(BayesianCoverage, NoiseAwareCoverage):
    """
    FUTURE WORK (design §4). Soft generalization of NoiseAwareCoverage: instead
    of a hard clean/noisy mask, down-weight each labeled point's coverage by its
    posterior reliability r_l (graph-based or AUM). Reuses NPC's inner-batch /
    train-between-batches loop, replacing the {0,1} mask with a soft r_l.

    NoiseAwareCoverage is the limiting case r_l = 1[AUM clean].
    """

    def __init__(self, cfg, annotator_manager=None, noise_filter=None):
        raise NotImplementedError(
            "NoiseAwareBayesianCoverage is future work (design §4): posterior "
            "reliability r_l via graph-based leave-one-out (preferred first) or "
            "AUM. Build BayesianCoverage + the product/additive comparison first."
        )
