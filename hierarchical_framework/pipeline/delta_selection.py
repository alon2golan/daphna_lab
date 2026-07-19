# How ProbCover's delta (the coverage-ball radius) gets chosen. We do NOT reinvent
# the rule: we delegate to deep_al's resolve_initial_delta -- the single source of
# truth also used by deep_al/pycls/al/prob_cover.py -- so our delta matches the
# paper's pipeline and prints the same [delta] banner saying which value was used.
#
# The chain resolve_initial_delta applies to --delta:
#   a positive float  -> use it as-is
#   "knee_delta"      -> run deep_al's data-driven knee-point algorithm on the features
#   "auto_delta"      -> look up the pinned (dataset, eval_mode) table
#                        (PROBCOVER_AUTO_DELTAS; ('CIFAR100','NN1') -> 0.65) and, if the
#                        row is missing (e.g. tinyimagenet), fall back to the knee.
# "auto_delta" is the default, matching deep_al.

import sys
from pathlib import Path
from types import SimpleNamespace

# deep_al lives at the repo root (hierarchical_framework/pipeline/ -> up three to daphna_lab/)
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DELTA_KEYWORDS = ("auto_delta", "knee_delta")  # the two non-numeric --delta modes deep_al accepts


def parse_delta_arg(delta_arg):
    # mirror deep_al's CLI parser: a known keyword stays a string, anything else is a float
    if delta_arg in DELTA_KEYWORDS:
        return delta_arg
    return float(delta_arg)


def resolve_delta(delta_arg, features, num_classes, seed=1,
                  dataset="cifar100", eval_mode="NN1_normalized"):
    # resolve_initial_delta reads three cfg fields; we don't build deep_al's full
    # training cfg, so hand it a minimal stand-in carrying just those fields.
    from deep_al.pycls.al.delta_defaults import resolve_initial_delta, PROBCOVER_AUTO_DELTAS
    cfg = SimpleNamespace(
        ACTIVE_LEARNING=SimpleNamespace(INITIAL_DELTA=parse_delta_arg(delta_arg)),  # a float or a keyword
        DATASET=SimpleNamespace(NAME=dataset.upper()),  # table is keyed by deep_al's name, e.g. 'CIFAR100'
        EVAL_MODEL_TYPE=eval_mode,                       # canonicalized to 'NN1' for the table lookup
    )
    # ProbCover's pinned table; method_name only labels the [delta] banner it prints.
    # Returns the numeric delta and logs which mode/value was chosen.
    return resolve_initial_delta(cfg, features, num_classes, seed,
                                 auto_table=PROBCOVER_AUTO_DELTAS, method_name="ProbCover")
