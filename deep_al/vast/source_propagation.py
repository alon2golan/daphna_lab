"""Author recurrence with the lab's initial-plateau guard and replay observer.

Messages and frozen-row updates call the unchanged public author function.
The surrounding stopping rule ports the lab source (2026-09-29), lines 746-759.
This is an experimental source-based protocol, not confirmed paper reproduction.
"""
import numpy as np

from . import author_reference
from .inference import normalize_evidence


def propagate_source(kernel, evidence, labeled_mask, labeled_indices, labeled_labels,
                     max_rounds=50, sender_threshold=.05, receiver_threshold=.75,
                     observer=None):
    if type(max_rounds) is not int or max_rounds < 1:
        raise ValueError("Source propagation needs a positive integer round cap")
    if not (np.isfinite(sender_threshold) and 0 <= sender_threshold < 1
            and np.isfinite(receiver_threshold) and 0 < receiver_threshold <= 1):
        raise ValueError("Invalid source sender/receiver thresholds")
    current = np.asarray(evidence, dtype=np.float64).copy()
    labeled_mask = np.asarray(labeled_mask, dtype=bool)
    alpha = 1 / current.shape[1]
    q = normalize_evidence(current)
    frozen = labeled_mask | (q.max(axis=1) > receiver_threshold)
    if observer is not None:
        empty = np.zeros(len(current), dtype=bool)
        observer(0, q, empty, empty, frozen, empty)
    history = []
    previous_receivers = None
    initial_count = None
    previous_count = None
    receivers_decreased = False
    termination = "max_rounds"
    for iteration in range(max_rounds):
        senders = author_reference.compute_sender_mask(
            current, labeled_mask, mode="high_confidence", threshold=sender_threshold)
        receivers = author_reference.compute_receiver_mask(
            current, labeled_mask, mode="high_confidence", threshold=receiver_threshold,
            prev_mask=previous_receivers)
        count = int(receivers.sum())
        if initial_count is None:
            initial_count = count
        elif count < initial_count:
            receivers_decreased = True
        # Match lab early stopping BEFORE an update. Equal starting counts
        # alone do not stop evidence from spreading beyond the seed neighbors.
        if receivers_decreased and previous_count is not None and count == previous_count:
            termination = "receiver_plateau_after_initial_decrease"
            break
        previous_count = count
        updated = author_reference.propagate_labels(
            current, kernel, labeled_mask, labeled_indices, labeled_labels, alpha,
            num_iterations=1, sender_mode="high_confidence", sender_threshold=sender_threshold,
            receiver_mode="high_confidence", receiver_threshold=receiver_threshold,
            normalization="sum", decay_rate=1.0)
        new_q = normalize_evidence(updated)
        new_frozen = frozen | ((new_q.max(axis=1) > receiver_threshold) & ~labeled_mask)
        newly_frozen = new_frozen & ~frozen
        if observer is not None:
            observer(iteration + 1, new_q, senders, receivers, new_frozen, newly_frozen)
        history.append({"round": iteration + 1, "senders": int(senders.sum()),
                        "receivers_before": count, "newly_frozen": int(newly_frozen.sum()),
                        "max_confidence": float(new_q.max()),
                        "max_unlabeled_confidence": float(new_q[~labeled_mask].max())
                        if (~labeled_mask).any() else None})
        current, q, frozen = updated, new_q, new_frozen
        previous_receivers = receivers
    return current, {"mode": "author-source", "confirmed_reproduction": False,
                     "termination": termination, "rounds": len(history), "history": history,
                     "round_cap": max_rounds, "cap_reached": termination == "max_rounds",
                     "sender_threshold": sender_threshold, "receiver_threshold": receiver_threshold,
                     "receiver_freezes_when": "confidence > receiver_threshold",
                     "prior": alpha, "message": "V - prior", "decay_rate": 1.0,
                     "include_labeled_soft_senders": False,
                     "source_commit": author_reference.SOURCE_COMMIT,
                     "stop_rule": "initial receiver plateau allowed; stop on plateau after decrease"}
