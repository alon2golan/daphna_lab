# Thin wrapper around deep_al's data-driven knee-point delta selector, so
# run_al.py / accuracy_vs_budget.py can resolve --delta knee the same way deep_al's
# resolve_initial_delta does (KMeans pseudo-labels -> purity(delta) curve -> knee).

import sys
from pathlib import Path

# deep_al lives at the repo root (hierarchical_framework/pipeline/ -> up three to daphna_lab/)
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def resolve_delta(delta_arg, features, num_classes, seed=1):
    # a plain float string ("0.4") is used as-is; the keyword "knee" runs the
    # deep_al knee algorithm on these features and returns its delta.
    if str(delta_arg).lower() in ("knee", "knee_delta"):
        from deep_al.pycls.datasets.utils.select_delta import select_delta
        delta = float(select_delta(features=features, num_classes=num_classes, seed=seed))
        print(f"[delta] knee -> {delta:.4f} (deep_al select_delta on "
              f"{features.shape[0]} feats, {num_classes} classes)", flush=True)
        return delta
    return float(delta_arg)
