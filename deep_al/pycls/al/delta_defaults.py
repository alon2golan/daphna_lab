"""Resolves the initial delta for ProbCover / DCoM under one rule, used by
all three call sites (prob_cover.py, coverage_matrix_methods.py, DCoM.py).

Single CLI knob: --initial_delta. The user picks one of three modes (see
_initial_delta_arg in deep_al/tools/train_al.py):

    positive float      -> use this value as-is. END.
    'knee_delta'        -> always run the data-driven knee-point algorithm
                           (ds_utils.select_delta on a purity curve). END.
    'auto_delta'        -> look up the pinned (dataset, eval_mode) table;
                           fall back to the knee algorithm with a warning
                           if the (dataset, method, eval_mode) row is
                           missing. This is also the default when
                           --initial_delta is omitted.

A negative or otherwise invalid explicit float (e.g. --initial_delta -1) is
treated as a malformed explicit; it routes to the auto_delta mode with a
hint line explaining the fall-through. This is the same behavior as
omitting --initial_delta entirely.

The pinned tables live in this module. They are intentionally separate
per method: ProbCover and DCoM have different optimal deltas for the
same (dataset, eval_mode), so do not share a single table.

To pin a new value: add a row to the appropriate table below. Key by
(DATASET.NAME, canonical eval_mode) where canonical eval_mode is the
return of deep_al.pycls.core.config.canonical_eval_mode (which folds
NN1_normalized into NN1).
"""

import time

import deep_al.pycls.datasets.utils as ds_utils
from deep_al.pycls.core.config import canonical_eval_mode


# (DATASET.NAME, canonical eval_mode) -> delta. Add rows as you pin them.
PROBCOVER_AUTO_DELTAS = {
    ('CIFAR100', 'NN1'):         0.65,
    ('CIFAR100', 'from_images'): 0.75,
}

# DCoM pinned values are not yet defined. Until rows are added, all
# 'auto_delta' DCoM runs fall through to the knee-point algorithm with
# the warning text below.
DCOM_AUTO_DELTAS = {}


def _dataset_eval_key(cfg):
    dataset = cfg.DATASET.NAME
    eval_mode = canonical_eval_mode(getattr(cfg, 'EVAL_MODEL_TYPE', None))
    return dataset, eval_mode


def _run_knee(cfg, features, num_classes, seed):
    t0 = time.time()
    delta = ds_utils.select_delta(
        features=features,
        num_classes=num_classes,
        seed=seed,
    )
    elapsed = time.time() - t0
    return float(delta), elapsed


def resolve_initial_delta(cfg, features, num_classes, seed,
                          auto_table, method_name):
    """Return the delta to use, printing exactly one [delta] banner.

    Args:
        cfg: the training cfg. Reads cfg.ACTIVE_LEARNING.INITIAL_DELTA,
            cfg.DATASET.NAME, cfg.EVAL_MODEL_TYPE.
        features: 2D feature matrix passed to the knee algorithm.
        num_classes: passed to the knee algorithm (KMeans k).
        seed: passed to the knee algorithm.
        auto_table: dict keyed by (dataset, canonical eval_mode) -> float.
            Use PROBCOVER_AUTO_DELTAS for ProbCover, DCOM_AUTO_DELTAS for DCoM.
        method_name: 'ProbCover' or 'DCoM', used only in log lines.
    """
    raw = cfg.ACTIVE_LEARNING.INITIAL_DELTA

    # Mode EXPLICIT-FLOAT: user gave a usable positive number.
    if isinstance(raw, (int, float)) and raw >= 0:
        delta = float(raw)
        print(f"[delta] {method_name}: using EXPLICIT delta={delta:.4f} "
              f"(from --initial_delta).")
        return delta

    # Mode KNEE: user explicitly asked for the data-driven knee algorithm.
    if raw == 'knee_delta':
        delta, elapsed = _run_knee(cfg, features, num_classes, seed)
        print(f"[delta] {method_name}: using KNEE delta -> {delta:.4f} "
              f"(--initial_delta knee_delta, {elapsed:.2f}s).")
        return delta

    # Bad explicit (e.g. --initial_delta -1): not a valid positive float, not
    # a known keyword. Route to AUTO with a hint instead of silently using it.
    if isinstance(raw, (int, float)):
        print(f"[delta] {method_name}: --initial_delta={raw} is not a valid "
              f"positive float; routing to auto_delta.")

    # Mode AUTO: pinned table lookup; if (dataset, eval_mode) is missing,
    # fall back to the knee algorithm with a warning.
    table_label = (
        'PROBCOVER_AUTO_DELTAS' if auto_table is PROBCOVER_AUTO_DELTAS
        else 'DCOM_AUTO_DELTAS' if auto_table is DCOM_AUTO_DELTAS
        else f'{method_name}_AUTO_DELTAS'
    )
    key = _dataset_eval_key(cfg)
    if key in auto_table:
        delta = float(auto_table[key])
        print(f"[delta] {method_name}: using AUTO delta={delta:.4f} "
              f"for {key} from {table_label}. "
              f"Pass --initial_delta <float> to override, "
              f"or --initial_delta knee_delta to run the knee algorithm.")
        return delta

    # Table miss -> knee fallback. DCoM gets an extra hint about the empty table.
    empty_hint = ''
    if auto_table is DCOM_AUTO_DELTAS and not DCOM_AUTO_DELTAS:
        empty_hint = (
            f" (note: {table_label} is empty -- DCoM pinned values "
            f"are not yet defined; add rows when known.)"
        )
    delta, elapsed = _run_knee(cfg, features, num_classes, seed)
    print(f"[delta] {method_name}: AUTO delta requested but no row for "
          f"{key} in {table_label}. Falling back to KNEE -> {delta:.4f} "
          f"({elapsed:.2f}s).{empty_hint} "
          f"Add a row to {table_label} to suppress this warning.")
    return delta
