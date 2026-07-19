# The evaluators package: base machinery (base.py), the flat floors/ceilings (flat.py), and
# the two-stage hierarchical methods (hierarchical.py). Everything the rest of the pipeline
# needs is re-exported here, and EVALUATORS maps each --eval name to its class.

from .evaluator_base import Evaluator, HierarchicalEvaluator
from .flat_classifiers import OneNearestNeighbourEvaluator, MLPEvaluator
from .hierarchical_classifiers import (
    HierarchicalKnnEvaluator,
    HierarchicalKnnMarginalEvaluator,
    HierarchicalMlpRoutedEvaluator,
    HierarchicalMlpMarginalEvaluator,
    HierarchicalMlpLogitsEvaluator,
)

# name -> class, so --eval / the comparison script can pick a base classifier by string. The
# superclass oracle is a separate flag (--oracle) that composes with the two flat classifiers.
EVALUATORS = {
    evaluator_class.name: evaluator_class
    for evaluator_class in (
        OneNearestNeighbourEvaluator,
        MLPEvaluator,
        HierarchicalKnnEvaluator,
        HierarchicalKnnMarginalEvaluator,
        HierarchicalMlpRoutedEvaluator,
        HierarchicalMlpMarginalEvaluator,
        HierarchicalMlpLogitsEvaluator,
    )
}
