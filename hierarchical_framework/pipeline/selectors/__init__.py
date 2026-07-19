# The selectors package: shared bookkeeping (base.py), ProbCover (probcover.py), and the
# baseline strategies (baselines.py). SELECTORS maps each --strategy name to its class.

from .base import Selector, CoverageSelector, build_graph
from .probcover import ProbCoverSelector
from .baselines import RandomSelector, AllLabeledSelector

# name -> class, so the CLI / comparison script can pick a strategy by string.
SELECTORS = {
    selector_class.name: selector_class
    for selector_class in (ProbCoverSelector, RandomSelector, AllLabeledSelector)
}
