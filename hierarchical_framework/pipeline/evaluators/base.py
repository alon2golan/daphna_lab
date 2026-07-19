# Evaluator base classes + the model factories every evaluator reuses.
#
# An Evaluator takes the labeled indices each round and returns a test accuracy. Two
# independent axes: the base classifier (flat / hierarchical) and the superclass oracle
# (--oracle, restrict each prediction to the TRUE superclass = the family ceiling).
# Here: the Evaluator base, the HierarchicalEvaluator mixin, and shared MLP/kNN factories.

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Every MLP head in the pipeline (flat and hierarchical) is this exact architecture, so
# the only thing that changes between methods is HOW the heads are wired together.
MLP_HIDDEN_LAYER_SIZES = (256,)   # one hidden layer of 256 units
MLP_MAX_ITERATIONS = 300          # training epochs cap for each head

# Default neighbour count for the kNN superclass vote (stage A of the hierarchical kNN).
KNN_SUPERCLASS_NEIGHBOURS = 10


def make_mlp(seed):
    # One MLP head, built identically everywhere so results are comparable across methods.
    return MLPClassifier(hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES, max_iter=MLP_MAX_ITERATIONS,
                         random_state=seed)


def make_knn(n_neighbors):
    # One kNN classifier (features are already L2-normalized, so plain Euclidean = cosine order).
    return KNeighborsClassifier(n_neighbors=n_neighbors)


def full_class_probabilities(classifier, features, num_columns):
    # sklearn's predict_proba only has columns for the classes it actually SAW while training
    # (classifier.classes_). Scatter those back into a full (num_points, num_columns) matrix so
    # class id == column index -- unseen classes stay at probability 0. Used by the marginal
    # methods, which need per-class probabilities laid out over the whole label space.
    seen_probabilities = classifier.predict_proba(features)          # (num_points, num_seen_classes)
    full = np.zeros((features.shape[0], num_columns), dtype=np.float64)
    full[:, classifier.classes_] = seen_probabilities                # place each seen class in its own column
    return full


def report_unresolved(unresolved_count, test_count):
    # When a test point's true superclass has NO labeled example, the oracle has nothing to
    # answer with. We leave those points unresolved (a miss) and say so, rather than quietly
    # answering them from some other superclass.
    if unresolved_count:
        print(
            f"[oracle] {unresolved_count}/{test_count} test points had no labeled example "
            f"in their true superclass -> left unresolved (counted as wrong)",
            flush=True,
        )


class Evaluator:

    name = "base"

    def __init__(self, train_features, train_labels, test_features, test_labels, coarse_map=None, seed=0):
        self.train_features = train_features   # pool features (selection happens over these)
        self.train_labels = train_labels       # pool fine labels
        self.test_features = test_features     # held-out features, only ever used to score
        self.test_labels = test_labels         # held-out fine labels
        self.coarse_map = coarse_map           # fine label -> superclass id; hierarchical + oracle read this
        self.seed = seed
        self.num_classes = int(train_labels.max()) + 1   # number of fine classes (100 for CIFAR-100)

    def fit(self, labeled_indices):
        # Train and return an sklearn classifier on the labeled subset. Subclasses pick the model.
        raise NotImplementedError

    def predict(self, labeled_indices):
        # Plain prediction: the trained classifier labels every test point.
        classifier = self.fit(labeled_indices)
        return classifier.predict(self.test_features)

    def predict_within_true_superclass(self, labeled_indices):
        # Oracle prediction: each test point's answer is restricted to its TRUE superclass.
        # Each subclass implements the restriction in whatever way suits its model.
        raise NotImplementedError

    def _predictions(self, labeled_indices, use_superclass_oracle):
        # The predictions accuracies are measured from -- plain, or restricted to the
        # true superclass when the oracle is on.
        if use_superclass_oracle:
            if self.coarse_map is None:
                raise SystemExit(
                    "--oracle needs a coarse map (coarse_map.npy alongside the embeddings, "
                    "or a built-in dataset like cifar100)"
                )
            return self.predict_within_true_superclass(labeled_indices)
        return self.predict(labeled_indices)

    def accuracy(self, labeled_indices, use_superclass_oracle=False):
        # Percent of test points whose fine class is predicted correctly.
        predictions = self._predictions(labeled_indices, use_superclass_oracle)
        return 100.0 * float((predictions == self.test_labels).mean())

    def fine_and_coarse_accuracy(self, labeled_indices, use_superclass_oracle=False):
        # Fine accuracy AND superclass (coarse) accuracy from a SINGLE prediction pass.
        # Coarse accuracy = did the prediction land in the RIGHT superclass?
        if self.coarse_map is None:
            raise SystemExit(
                "superclass accuracy needs a coarse map (coarse_map.npy alongside the "
                "embeddings, or a built-in dataset like cifar100)"
            )
        predictions = self._predictions(labeled_indices, use_superclass_oracle)
        fine_accuracy = 100.0 * float((predictions == self.test_labels).mean())

        # Map predictions and truth up to superclasses. Unresolved oracle predictions (-1)
        # can never be coarse-correct, so keep them as an impossible superclass id.
        is_resolved = predictions >= 0
        predicted_superclasses = np.full(len(predictions), -1, dtype=np.int64)
        predicted_superclasses[is_resolved] = self.coarse_map[predictions[is_resolved]]
        true_superclasses = self.coarse_map[self.test_labels]
        coarse_accuracy = 100.0 * float((predicted_superclasses == true_superclasses).mean())

        return fine_accuracy, coarse_accuracy


class HierarchicalEvaluator(Evaluator):
    # Shared plumbing for the two-stage (superclass -> subclass) methods. Every hierarchical
    # method PREDICTS the superclass from features first (it never sees the test point's true
    # superclass -- that is the oracle's job, and the oracle is the separate family ceiling).

    name = "hierarchical-base"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.coarse_map is None:
            raise SystemExit(
                f"evaluator '{self.name}' is hierarchical and needs a coarse map "
                "(coarse_map.npy alongside the embeddings, or a built-in dataset like cifar100)"
            )
        self.num_superclasses = int(self.coarse_map.max()) + 1   # 20 for CIFAR-100

    def labeled_arrays(self, labeled_indices):
        # The three things every stage trains on: each labeled point's features, its fine
        # label, and the superclass that fine label rolls up to.
        labeled_features = self.train_features[labeled_indices]
        labeled_fine_labels = self.train_labels[labeled_indices]
        labeled_superclasses = self.coarse_map[labeled_fine_labels]
        return labeled_features, labeled_fine_labels, labeled_superclasses
