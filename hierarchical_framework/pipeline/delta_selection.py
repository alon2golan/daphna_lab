# Handles finding the correct coverage-ball radius (delta) using deep_al's logic.

import sys
from pathlib import Path
from types import SimpleNamespace

# Add repo root to path so we can import from the core deep_al package
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DELTA_KEYWORDS = ("auto_delta", "knee_delta")


def parse_delta_arg(delta_arg):
    """Converts the CLI string argument into either a keyword string or a raw float."""
    if delta_arg in DELTA_KEYWORDS:
        return delta_arg
    return float(delta_arg)


def resolve_delta(delta_arg, features, num_classes, seed=1,
                  dataset="cifar100", eval_mode="NN1_normalized"):
    """
    Delegates delta selection to deep_al's official resolve_initial_delta function.

    Creates a minimal configuration mockup to mimic the official deep_al environment,
    ensuring our pipeline reproduces the exact table lookups or knee-point math.
    """
    from deep_al.pycls.al.delta_defaults import resolve_initial_delta, PROBCOVER_AUTO_DELTAS

    # Mock deep_al's nested configuration system using simple namespaces
    cfg = SimpleNamespace(
        ACTIVE_LEARNING=SimpleNamespace(INITIAL_DELTA=parse_delta_arg(delta_arg)),
        DATASET=SimpleNamespace(NAME=dataset.upper()),
        EVAL_MODEL_TYPE=eval_mode,
    )

    # Run the official resolver tool using the default deep_al auto-lookup table
    return resolve_initial_delta(cfg, features, num_classes, seed,
                                 auto_table=PROBCOVER_AUTO_DELTAS, method_name="ProbCover")