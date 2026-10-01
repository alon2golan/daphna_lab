"""Exact author propagation helpers for source-parity diagnostics.

Copied from deep-al/tools/train_al.py at author commit
3d4cbd7d543b1d1290799bdfb02687519af3a264. See LICENSE.upstream.
These functions differ from the written paper recurrence. They are not the
production reproduction default and do not identify the paper experiment config.
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F

SOURCE_COMMIT = "3d4cbd7d543b1d1290799bdfb02687519af3a264"
SOURCE_PATH = "deep-al/tools/train_al.py"
logger = logging.getLogger(__name__)

def normalize_C(C, normalization='sum'):
    """Row-normalize a (N, num_classes) numpy confidence matrix.

    Args:
        C: numpy array (N, num_classes)
        normalization: 'sum' — divide each row by its sum; 'softmax' — torch softmax
    Returns:
        numpy float array (N, num_classes); zero rows map to zero
    """
    if normalization == 'softmax':
        return F.softmax(torch.from_numpy(C.astype(np.float32)), dim=1).numpy()
    row_sums = C.sum(axis=1, keepdims=True)
    return np.divide(C, row_sums, out=np.zeros_like(C, dtype=float), where=row_sums != 0)


def compute_sender_mask(C, labeled_mask, mode='all', threshold=None,
                        al_indices=None, al_resample_fn=None, normalization='sum', sender_budget=None):
    """Which unlabeled points may propagate (send) information this iteration.

    Labeled points are always excluded.  Sender modes:
        'all'             – every unlabeled point sends
        'high_confidence' – unlabeled with norm_C.max > threshold  (threshold #1)
                            If sender_budget is specified, select top K points by max(C) instead
        'al_constant'     – the AL picks from the last round, fixed for all iterations
        'al_updated'      – call al_resample_fn(C) each iteration to re-select via bayes_misp

    Note: 'al_constant' and 'al_updated' are equivalent when num_iterations=1, since
    sender selection happens only once regardless.

    Args:
        C: confidence matrix (N, num_classes)
        labeled_mask: boolean mask (N,) indicating labeled points
        mode: sender mode ('all', 'high_confidence', 'al_constant', 'al_updated')
        threshold: confidence threshold for 'high_confidence' mode
        al_indices: pre-selected indices for 'al_constant' mode
        al_resample_fn: callable for 'al_updated' mode
        normalization: 'sum' or 'softmax'
        sender_budget: optional int, number of senders to select (for 'high_confidence' mode)

    Returns boolean mask (N,).
    """
    sender_mask = ~labeled_mask  # base: all unlabeled

    if mode == 'high_confidence':
        norm = normalize_C(C, normalization)
        max_conf = norm.max(axis=1)
        
        if sender_budget is not None:
            # Top K by confidence
            unlabeled_indices = np.where(sender_mask)[0]
            if len(unlabeled_indices) > sender_budget:
                unlabeled_conf = max_conf[unlabeled_indices]
                top_k_local = np.argsort(unlabeled_conf)[-sender_budget:]
                top_k_global = unlabeled_indices[top_k_local]
                new_mask = np.zeros_like(sender_mask)
                new_mask[top_k_global] = True
                sender_mask = new_mask
            # else: use all unlabeled (budget >= available points)
        else:
            # Original threshold behavior
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
                         al_indices=None, al_resample_fn=None, normalization='sum', receiver_budget=None):
    """Which points may receive propagated information this iteration.

    Labeled points never receive.  Receiver modes:
        'all'             – every unlabeled point can receive (default)
        'high_confidence' – unlabeled with norm_C.max <= threshold can receive (low confidence)
                            Once a point crosses threshold it is frozen for remaining iterations
                            If receiver_budget is specified, select bottom K points by max(C) instead
        'al_constant'     – the AL picks from the last round, fixed for all iterations
        'al_updated'      – call al_resample_fn(C) each iteration to re-select via bayes_misp

    Args:
        C:         current C matrix (N, num_classes) numpy array
        labeled_mask: boolean (N,)
        mode: receiver mode ('all', 'high_confidence', 'al_constant', 'al_updated')
        threshold: threshold #2; norm_C.max > threshold → point is frozen (for 'high_confidence' mode)
        prev_mask: receiver mask from the previous iteration (None = initialise fresh)
        al_indices: pre-selected indices for 'al_constant' mode
        al_resample_fn: callable for 'al_updated' mode
        normalization: passed to normalize_C
        receiver_budget: optional int, number of receivers to select (for 'high_confidence' mode)

    Returns boolean mask (N,).
    """
    if prev_mask is None:
        prev_mask = ~labeled_mask  # initial: all unlabeled can receive
    
    if mode == 'all':
        # No filtering, all unlabeled can receive
        return prev_mask
    
    elif mode == 'high_confidence':
        # Points with LOW confidence can receive (below threshold)
        # Once a point crosses threshold it is frozen
        if threshold is None:
            return prev_mask
        
        norm = normalize_C(C, normalization)
        max_conf = norm.max(axis=1)
        
        if receiver_budget is not None:
            # Select bottom K points by confidence (most uncertain)
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
            # Original threshold behavior: freeze points that exceed threshold
            exceeded = max_conf > threshold
            return prev_mask & ~exceeded  # once frozen, always frozen
    
    elif mode == 'al_constant':
        # Only AL-selected points can receive
        if al_indices is not None and len(al_indices) > 0:
            al_mask = np.zeros(len(labeled_mask), dtype=bool)
            al_mask[np.asarray(al_indices, dtype=int)] = True
            return prev_mask & al_mask
        else:
            print('[Propagation] Warning: al_constant receiver mode but no al_indices — falling back to all unlabeled')
            return prev_mask
    
    elif mode == 'al_updated':
        # Re-run AL selection each iteration
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
                     receiver_threshold=None, receiver_mode='all', receiver_al_indices=None, 
                     receiver_al_resample_fn=None, receiver_budget=None,
                     al_indices=None, al_resample_fn=None,
                     normalization='sum', sender_budget=None, low_rank=None, decay_rate=1.0):
    """Propagate label information from source points to their K-weighted neighbors.

    Update rule per iteration, per point j, per class m:
        V_new[j, m] = α
                    + Σ_{s∈sources} K[s,j] · 1_{y_s=m}       (labeled contribution)
                    + Σ_{u∈can_send} K[u,j] · (C[u,m] − α)   (unlabeled propagation)

    At every iteration both masks are recomputed from current C:
      • can_send_mask    – determined by sender_mode / sender_threshold / sender_budget
      • can_receive_mask – determined by receiver_mode / receiver_threshold / receiver_budget

    Source (labeled / oracle) rows are always restored to their original values.

    Args:
        C_matrix:           (N, num_classes) confidence matrix — torch tensor or numpy
        K_matrix:           (N, N) kernel — sparse CSR, torch tensor, or numpy
        labeled_mask:       boolean (N,) — which points are labeled
        source_indices:     int array — fixed information sources (labeled or oracle)
        source_labels:      int array — class label per source point
        alpha:              baseline scalar
        num_iterations:     number of propagation rounds
        sender_mode:        'all' | 'high_confidence' | 'al_constant' | 'al_updated'
        sender_threshold:   threshold #1 for 'high_confidence' sender mode
        receiver_threshold: threshold #2 for 'high_confidence' receiver mode
        receiver_mode:      'all' | 'high_confidence' | 'al_constant' | 'al_updated'
        receiver_al_indices:     pre-selected indices for 'al_constant' receiver mode
        receiver_al_resample_fn: callable(C) -> indices for 'al_updated' receiver mode
        receiver_budget:    optional int, number of receivers (K) for 'high_confidence' mode
        al_indices:         pre-selected indices for 'al_constant' sender mode
        al_resample_fn:     callable(C) -> indices for 'al_updated' sender mode
        normalization:      'sum' | 'softmax'
        sender_budget:      optional int, number of senders (K) for 'high_confidence' mode
        low_rank:           optional int — when K is dense and GPU is unavailable, approximate
                            K ≈ Q·diag(λ)·Qᵀ using the top-`low_rank` eigenvectors (precomputed
                            once via scipy eigsh).  Each multiply becomes O(N·r·C) instead of
                            O(N²·C).  Ignored when K is sparse or CUDA is available.
        decay_rate:         float in [0, 1] — decay factor per iteration for unlabeled propagation.
                            At iteration i, the unlabeled propagation is scaled by decay_rate^i.
                            Default 1.0 (no decay). Values < 1.0 make later iterations propagate
                            less information from unlabeled points.

    Returns:
        C matrix (same type and device as input) after propagation.
    """
    import scipy.sparse as sp

    # --- type bookkeeping (round-trip tensor ↔ numpy) -------------------------
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
        K_obj = np.asarray(K_obj, dtype=np.float32)  # float32 BLAS is ~2x faster than float64

        # K is too large to fit in VRAM entirely; multiply in row-chunks so only
        # one tile (chunk_size × |active|) lives on GPU at a time.
        _K_gpu_chunk_size = 512   # rows per tile; tune down if still OOM
        print(f'[Propagation] Dense K backend: GPU chunked (chunk={_K_gpu_chunk_size}, '
              f'K={K_obj.nbytes / 1e9:.2f} GB on CPU)')
        logger.info(f'[Propagation] Dense K backend: GPU chunked (chunk={_K_gpu_chunk_size})')

        def _K_mul(X, row_mask=None):
            active = np.where(row_mask)[0] if (row_mask is not None and row_mask.sum() < len(row_mask)) else None
            X_src = X[active] if active is not None else X          # (|active|, C) or (N, C)
            K_cols = K_obj[:, active] if active is not None else K_obj   # (N, |active|) cpu
            X_t = torch.from_numpy(X_src.astype(np.float32)).cuda()     # (|active|, C) on GPU

            out = np.empty((K_cols.shape[0], X_src.shape[1]), dtype=np.float32)
            for start in range(0, K_cols.shape[0], _K_gpu_chunk_size):
                end = min(start + _K_gpu_chunk_size, K_cols.shape[0])
                K_tile = torch.from_numpy(K_cols[start:end]).cuda()     # (chunk, |active|)
                out[start:end] = (K_tile @ X_t).cpu().numpy()
            return out

        def _K_row_sum(idx): return K_obj[idx, :].sum(axis=0)

    # --- precompute labeled contribution (constant across iterations) ---------
    # labeled_contribution[j, m] = Σ_{s : y_s=m} K[s, j]
    labeled_contribution = np.zeros((n_samples, num_classes), dtype=np.float32)
    for cls in range(num_classes):
        src_cls = source_indices[source_labels == cls]
        if len(src_cls) > 0:
            labeled_contribution[:, cls] = _K_row_sum(src_cls)

    original_source_values = C_current[labeled_indices].copy()

    print(f'[Propagation] sources={len(source_indices)}, sender_mode={sender_mode!r}, '
          f'receiver_mode={receiver_mode!r}, iterations={num_iterations}, '
          f'sender_threshold={sender_threshold}, sender_budget={sender_budget}, '
          f'receiver_threshold={receiver_threshold}, receiver_budget={receiver_budget}, '
          f'decay_rate={decay_rate}')
    logger.info(f'[Propagation] sources={len(source_indices)}, sender_mode={sender_mode!r}, '
                f'receiver_mode={receiver_mode!r}, iterations={num_iterations}, '
                f'sender_budget={sender_budget}, receiver_budget={receiver_budget}, '
                f'decay_rate={decay_rate}')

    # --- propagation loop -----------------------------------------------------
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

        # Compute decay factor for this iteration (decay_rate^i)
        decay_factor = decay_rate ** i
        
        print(f'[Propagation] iter {i + 1}/{num_iterations}: '
              f'senders={can_send.sum()}, receivers={can_receive.sum()}, decay_factor={decay_factor:.4f}')
        logger.info(f'[Propagation] iter {i + 1}/{num_iterations}: '
                    f'senders={can_send.sum()}, receivers={can_receive.sum()}, decay_factor={decay_factor:.4f}')

        V_excess = (C_current - alpha) * can_send[:, None]
        unlabeled_propagation = _K_mul(V_excess, row_mask=can_send)
        V_new = alpha + labeled_contribution + decay_factor * unlabeled_propagation

        V_new[~can_receive] = C_current[~can_receive]   # keep frozen values
        V_new[labeled_indices] = original_source_values  # restore sources

        C_current = V_new

    if is_tensor:
        return torch.from_numpy(C_current).to(original_device)
    return C_current
