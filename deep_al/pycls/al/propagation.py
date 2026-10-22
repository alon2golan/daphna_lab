"""
Label propagation over a kernel similarity matrix: spreads per-class evidence
from labeled (or otherwise trusted) points to their K-weighted neighbors, so an
evidence matrix C (e.g. BAYES_MISP's C_general) can be updated with information
from currently-unlabeled points before it's consumed downstream (e.g. as
distillation targets).

Not wired into the training loop (train_al.py) yet -- this module is
standalone and importable/testable on its own. A follow-up should attach it to
an actual training run.

Ported from Itai's TypiClust experimental workspace, trimmed to the threshold
('high_confidence') and active-learning-choice ('al_constant' / 'al_updated')
sender/receiver modes. Left out: the 'per_class' curriculum-threshold mode,
the 'otsu' data-driven threshold mode, per-iteration decay, and the low-rank
dense-K approximation -- all ablation-study variants layered on top of the
same core mechanism.
"""
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp


def normalize_C(C, normalization='sum'):
    """Row-normalize a (N, num_classes) numpy confidence matrix.

    Args:
        C: numpy array (N, num_classes)
        normalization: 'sum' -- divide each row by its sum; 'softmax' -- torch softmax
    Returns:
        numpy float array (N, num_classes); zero rows map to zero
    """
    if normalization == 'softmax':
        return F.softmax(torch.from_numpy(C.astype(np.float32)), dim=1).numpy()
    row_sums = C.sum(axis=1, keepdims=True)
    return np.divide(C, row_sums, out=np.zeros_like(C, dtype=float), where=row_sums != 0)


def compute_sender_mask(C, labeled_mask, mode='all', threshold=None,
                        al_indices=None, al_resample_fn=None, normalization='sum',
                        sender_budget=None):
    """Which unlabeled points may propagate (send) information this iteration.

    Labeled points are always excluded. Sender modes:
        'all'             -- every unlabeled point sends
        'high_confidence' -- unlabeled with norm_C.max > threshold.
                            If sender_budget is specified, select the top-K points
                            by max(C) instead of thresholding.
        'al_constant'     -- the AL algorithm's picks from the last round, fixed
                            for all iterations
        'al_updated'      -- call al_resample_fn(C) each iteration to re-select
                            senders via the AL algorithm (e.g. bayes_misp) on the
                            current C

    Note: 'al_constant' and 'al_updated' are equivalent when num_iterations=1,
    since sender selection happens only once regardless.

    Args:
        C: confidence matrix (N, num_classes)
        labeled_mask: boolean mask (N,) indicating labeled points
        mode: sender mode ('all', 'high_confidence', 'al_constant', 'al_updated')
        threshold: confidence threshold for 'high_confidence' mode
        al_indices: pre-selected indices for 'al_constant' mode
        al_resample_fn: callable(C) -> indices, for 'al_updated' mode
        normalization: 'sum' or 'softmax'
        sender_budget: optional int, number of senders to select (for 'high_confidence' mode)

    Returns boolean mask (N,).
    """
    sender_mask = ~labeled_mask  # base: all unlabeled

    if mode == 'high_confidence':
        norm = normalize_C(C, normalization)
        max_conf = norm.max(axis=1)

        if sender_budget is not None:
            unlabeled_indices = np.where(sender_mask)[0]
            if len(unlabeled_indices) > sender_budget:
                unlabeled_conf = max_conf[unlabeled_indices]
                top_k_local = np.argsort(unlabeled_conf)[-sender_budget:]
                top_k_global = unlabeled_indices[top_k_local]
                new_mask = np.zeros_like(sender_mask)
                new_mask[top_k_global] = True
                sender_mask = new_mask
            # else: budget >= available points, use all unlabeled
        else:
            sender_mask = sender_mask & (max_conf > threshold)

    elif mode == 'al_constant':
        if al_indices is not None and len(al_indices) > 0:
            al_mask = np.zeros(len(labeled_mask), dtype=bool)
            al_mask[np.asarray(al_indices, dtype=int)] = True
            sender_mask = sender_mask & al_mask
        else:
            print('[Propagation] Warning: al_constant mode but no al_indices — falling back to all unlabeled')

    elif mode == 'al_updated':
        if al_resample_fn is not None:
            new_indices = np.asarray(al_resample_fn(C), dtype=int)
            al_mask = np.zeros(len(labeled_mask), dtype=bool)
            al_mask[new_indices] = True
            sender_mask = sender_mask & al_mask
        else:
            print('[Propagation] Warning: al_updated mode but no al_resample_fn — falling back to all unlabeled')

    return sender_mask


def compute_receiver_mask(C, labeled_mask, mode='all', threshold=None, prev_mask=None,
                         al_indices=None, al_resample_fn=None, normalization='sum',
                         receiver_budget=None):
    """Which points may receive propagated information this iteration.

    Labeled points never receive. Receiver modes:
        'all'             -- every unlabeled point can receive (default)
        'high_confidence' -- unlabeled with norm_C.max <= threshold can receive
                            (low confidence). Once a point crosses threshold it
                            is frozen for remaining iterations. If
                            receiver_budget is specified, select the bottom-K
                            points by max(C) instead.
        'al_constant'     -- the AL algorithm's picks from the last round, fixed
                            for all iterations
        'al_updated'      -- call al_resample_fn(C) each iteration to re-select
                            receivers via the AL algorithm on the current C

    Args:
        C:         current C matrix (N, num_classes) numpy array
        labeled_mask: boolean (N,)
        mode: receiver mode ('all', 'high_confidence', 'al_constant', 'al_updated')
        threshold: norm_C.max > threshold -> point is frozen (for 'high_confidence' mode)
        prev_mask: receiver mask from the previous iteration (None = initialise fresh)
        al_indices: pre-selected indices for 'al_constant' mode
        al_resample_fn: callable(C) -> indices, for 'al_updated' mode
        normalization: passed to normalize_C
        receiver_budget: optional int, number of receivers to select (for 'high_confidence' mode)

    Returns boolean mask (N,).
    """
    if prev_mask is None:
        prev_mask = ~labeled_mask  # initial: all unlabeled can receive

    if mode == 'all':
        return prev_mask

    elif mode == 'high_confidence':
        if threshold is None:
            return prev_mask

        norm = normalize_C(C, normalization)
        max_conf = norm.max(axis=1)

        if receiver_budget is not None:
            unlabeled_indices = np.where(prev_mask)[0]
            if len(unlabeled_indices) > receiver_budget:
                unlabeled_conf = max_conf[unlabeled_indices]
                bottom_k_local = np.argsort(unlabeled_conf)[:receiver_budget]
                bottom_k_global = unlabeled_indices[bottom_k_local]
                new_mask = np.zeros_like(prev_mask)
                new_mask[bottom_k_global] = True
                return new_mask
            else:
                return prev_mask  # budget >= available points
        else:
            exceeded = max_conf > threshold
            return prev_mask & ~exceeded  # once frozen, always frozen

    elif mode == 'al_constant':
        if al_indices is not None and len(al_indices) > 0:
            al_mask = np.zeros(len(labeled_mask), dtype=bool)
            al_mask[np.asarray(al_indices, dtype=int)] = True
            return prev_mask & al_mask
        else:
            print('[Propagation] Warning: al_constant receiver mode but no al_indices — falling back to all unlabeled')
            return prev_mask

    elif mode == 'al_updated':
        if al_resample_fn is not None:
            new_indices = np.asarray(al_resample_fn(C), dtype=int)
            al_mask = np.zeros(len(labeled_mask), dtype=bool)
            al_mask[new_indices] = True
            return prev_mask & al_mask
        else:
            print('[Propagation] Warning: al_updated receiver mode but no al_resample_fn — falling back to all unlabeled')
            return prev_mask

    else:
        print(f'[Propagation] Warning: unknown receiver mode {mode!r} — falling back to all unlabeled')
        return prev_mask


def propagate_labels(C_matrix, K_matrix, labeled_mask, source_indices, source_labels, alpha,
                     num_iterations=1, sender_mode='all', sender_threshold=None,
                     sender_budget=None, receiver_mode='all', receiver_threshold=None,
                     receiver_budget=None, al_indices=None, al_resample_fn=None,
                     receiver_al_indices=None, receiver_al_resample_fn=None,
                     normalization='sum'):
    """Propagate label information from source points to their K-weighted neighbors.

    Update rule per iteration, per point j, per class m:
        V_new[j, m] = alpha
                    + sum_{s in sources} K[s,j] * 1_{y_s=m}       (labeled contribution)
                    + sum_{u in can_send} K[u,j] * (C[u,m] - alpha)  (unlabeled propagation)

    At every iteration both masks are recomputed from the current C:
      - can_send_mask    -- determined by sender_mode / sender_threshold / sender_budget
      - can_receive_mask -- determined by receiver_mode / receiver_threshold / receiver_budget

    Source (labeled / oracle) rows are always restored to their original values.

    Args:
        C_matrix:           (N, num_classes) confidence matrix -- torch tensor or numpy
        K_matrix:           (N, N) kernel -- sparse CSR, torch tensor, or numpy
        labeled_mask:       boolean (N,) -- which points are labeled
        source_indices:     int array -- fixed information sources (labeled or oracle)
        source_labels:      int array -- class label per source point
        alpha:              baseline scalar
        num_iterations:     number of propagation rounds
        sender_mode:        'all' | 'high_confidence' | 'al_constant' | 'al_updated'
        sender_threshold:   threshold for 'high_confidence' sender mode
        sender_budget:      optional int, number of senders (K) for 'high_confidence' mode
        receiver_mode:      'all' | 'high_confidence' | 'al_constant' | 'al_updated'
        receiver_threshold: threshold for 'high_confidence' receiver mode
        receiver_budget:    optional int, number of receivers (K) for 'high_confidence' mode
        al_indices:         pre-selected indices for 'al_constant' sender mode
        al_resample_fn:     callable(C) -> indices, for 'al_updated' sender mode
        receiver_al_indices:     pre-selected indices for 'al_constant' receiver mode
        receiver_al_resample_fn: callable(C) -> indices, for 'al_updated' receiver mode
        normalization:      'sum' | 'softmax'

    Returns:
        C matrix (same type and device as input) after propagation.
    """
    is_tensor = torch.is_tensor(C_matrix)
    original_device = C_matrix.device if is_tensor else 'cpu'
    C_current = C_matrix.cpu().numpy() if is_tensor else np.array(C_matrix, copy=True)

    labeled_mask = np.asarray(
        labeled_mask.cpu() if torch.is_tensor(labeled_mask) else labeled_mask, dtype=bool)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    source_labels = np.asarray(source_labels, dtype=np.int64)

    n_samples, num_classes = C_current.shape
    labeled_indices = np.where(labeled_mask)[0]

    # --- K: build unified multiply and row-sum helpers ------------------------
    if sp.issparse(K_matrix):
        K_obj = K_matrix
        def _K_mul(X, row_mask=None): return K_obj @ X
        def _K_row_sum(idx): return K_obj[idx, :].sum(axis=0).A1
    else:
        K_obj = K_matrix.cpu().numpy() if torch.is_tensor(K_matrix) else np.asarray(K_matrix)
        K_obj = np.asarray(K_obj, dtype=np.float32)

        def _K_mul(X, row_mask=None):
            return K_obj @ X

        def _K_row_sum(idx): return K_obj[idx, :].sum(axis=0)

    # --- precompute labeled contribution (constant across iterations) ---------
    # labeled_contribution[j, m] = sum_{s : y_s=m} K[s, j]
    labeled_contribution = np.zeros((n_samples, num_classes), dtype=np.float32)
    for cls in range(num_classes):
        src_cls = source_indices[source_labels == cls]
        if len(src_cls) > 0:
            labeled_contribution[:, cls] = _K_row_sum(src_cls)

    original_source_values = C_current[labeled_indices].copy()

    print(f'[Propagation] sources={len(source_indices)}, sender_mode={sender_mode!r}, '
          f'receiver_mode={receiver_mode!r}, iterations={num_iterations}, '
          f'sender_threshold={sender_threshold}, sender_budget={sender_budget}, '
          f'receiver_threshold={receiver_threshold}, receiver_budget={receiver_budget}')

    can_receive = None  # initialised on first call to compute_receiver_mask

    for i in range(num_iterations):
        can_send = compute_sender_mask(
            C_current, labeled_mask,
            mode=sender_mode, threshold=sender_threshold,
            al_indices=al_indices, al_resample_fn=al_resample_fn,
            normalization=normalization,
            sender_budget=sender_budget,
        )
        can_receive = compute_receiver_mask(
            C_current, labeled_mask,
            mode=receiver_mode, threshold=receiver_threshold, prev_mask=can_receive,
            al_indices=receiver_al_indices, al_resample_fn=receiver_al_resample_fn,
            normalization=normalization,
            receiver_budget=receiver_budget,
        )

        print(f'[Propagation] iter {i + 1}/{num_iterations}: '
              f'senders={can_send.sum()}, receivers={can_receive.sum()}')

        V_excess = (C_current - alpha) * can_send[:, None]
        unlabeled_propagation = _K_mul(V_excess, row_mask=can_send)
        V_new = alpha + labeled_contribution + unlabeled_propagation

        V_new[~can_receive] = C_current[~can_receive]   # keep frozen values
        V_new[labeled_indices] = original_source_values  # restore sources

        C_current = V_new

    if is_tensor:
        return torch.from_numpy(C_current).to(original_device)
    return C_current


def build_al_resample_fn(al_algorithm, cfg, lSet, uSet, budget=None):
    """
    Build a callable(C) -> indices that re-runs the AL algorithm's own
    sample_from_uSet on a candidate C matrix, for use as al_resample_fn in
    compute_sender_mask/compute_receiver_mask's 'al_updated' mode.

    Temporarily swaps the AL algorithm's C_general for the candidate C, runs
    selection, then restores all mutated state -- the real round is never
    affected by this "fake" AL call.

    Args:
        al_algorithm: the concrete AL strategy instance (e.g. BAYES_MISP),
            i.e. ActiveLearning.al_algorithm -- must expose C_general and
            sample_from_uSet(clf_model, lSet, uSet, trainDataset).
        cfg: config object (only ACTIVE_LEARNING.BUDGET_SIZE is read/restored).
        lSet, uSet: labeled/unlabeled indices to pass through to sample_from_uSet.
        budget: optional int overriding cfg.ACTIVE_LEARNING.BUDGET_SIZE / the
            algorithm's own budgetSize for the duration of this resample call.

    Returns:
        callable(C: np.ndarray) -> np.ndarray of selected indices.
    """
    lSet = np.asarray(lSet, dtype=np.int64)
    uSet = np.asarray(uSet, dtype=np.int64)

    def _resample_from_C(C):
        if not hasattr(al_algorithm, 'C_general'):
            print('[Propagation] al_updated: algorithm has no C_general — falling back to all unlabeled')
            return uSet

        saved_budget_cfg = cfg.ACTIVE_LEARNING.BUDGET_SIZE
        saved_budget_alg = getattr(al_algorithm, 'budgetSize', None)
        if budget is not None:
            cfg.ACTIVE_LEARNING.BUDGET_SIZE = budget
            if hasattr(al_algorithm, 'budgetSize'):
                al_algorithm.budgetSize = budget

        original_C_general = al_algorithm.C_general
        saved_labeled_mask = (
            al_algorithm.labeled_points_mask_general.clone()
            if hasattr(al_algorithm, 'labeled_points_mask_general') else None
        )
        saved_cum_labels_info = (
            al_algorithm.cum_labels_info.clone()
            if hasattr(al_algorithm, 'cum_labels_info') else None
        )

        al_algorithm.C_general = torch.from_numpy(C.astype(np.float32)).to(original_C_general.device)

        try:
            activeSet, _ = al_algorithm.sample_from_uSet(None, lSet, uSet, None)
            indices = np.asarray(activeSet, dtype=int)
        except Exception as e:
            print(f'[Propagation] al_updated: sample_from_uSet failed ({e}), falling back to all unlabeled')
            indices = uSet
        finally:
            cfg.ACTIVE_LEARNING.BUDGET_SIZE = saved_budget_cfg
            if saved_budget_alg is not None:
                al_algorithm.budgetSize = saved_budget_alg
            al_algorithm.C_general = original_C_general
            if saved_labeled_mask is not None:
                al_algorithm.labeled_points_mask_general = saved_labeled_mask
            if saved_cum_labels_info is not None:
                al_algorithm.cum_labels_info = saved_cum_labels_info

        return indices

    return _resample_from_C
