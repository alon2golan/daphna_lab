# Evaluation methods -- the overridable "how do we score a labeled set?".
#
# The AL loop hands an Evaluator the labeled indices each round and gets back a test
# accuracy (percent). There are two INDEPENDENT choices:
#
#   * the base classifier   -- 1nn or mlp        (the --eval flag / the EVALUATORS registry)
#   * the superclass oracle -- optionally restrict each prediction to the test point's
#                              TRUE superclass   (the --oracle flag)
#
# The oracle is base-method-agnostic: every base classifier defines how IT restricts a
# prediction to a superclass. 1-NN restricts the neighbour pool to that superclass; the
# MLP keeps only the output classes in that superclass. Add a new base classifier by
# subclassing Evaluator and registering it in EVALUATORS.

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def report_unresolved(unresolved_count, test_count):
    # When a test point's true superclass has NO labeled example, the oracle has nothing
    # to answer with. We leave those points unresolved (a miss) and say so out loud,
    # rather than quietly answering them from some other superclass.
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
        self.coarse_map = coarse_map           # fine label -> superclass id; only the oracle path reads this
        self.seed = seed

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


class OneNearestNeighbourEvaluator(Evaluator):

    name = "1nn"

    def fit(self, labeled_indices):
        # 1-nearest-neighbour on the labeled subset (features are already L2-normalized).
        classifier = KNeighborsClassifier(n_neighbors=1)
        classifier.fit(self.train_features[labeled_indices], self.train_labels[labeled_indices])
        return classifier

    def predict_within_true_superclass(self, labeled_indices):
        # For 1-NN, "superclass known" means: only search labeled points inside that superclass.
        labeled_features = self.train_features[labeled_indices]
        labeled_fine_labels = self.train_labels[labeled_indices]
        labeled_superclasses = self.coarse_map[labeled_fine_labels]
        test_superclasses = self.coarse_map[self.test_labels]

        # -1 marks an unresolved test point; it never equals a real label, so it counts as a miss.
        predictions = np.full(len(self.test_labels), -1, dtype=np.int64)
        unresolved_count = 0

        for superclass in np.unique(test_superclasses):
            test_in_superclass = test_superclasses == superclass          # the test points to answer this pass
            labeled_in_superclass = labeled_superclasses == superclass    # candidate neighbours in that superclass

            # No labeled example in this true superclass -> genuinely nothing to search.
            if not labeled_in_superclass.any():
                unresolved_count += int(test_in_superclass.sum())
                continue

            # 1-NN over only the same-superclass labeled points.
            classifier = KNeighborsClassifier(n_neighbors=1)
            classifier.fit(labeled_features[labeled_in_superclass], labeled_fine_labels[labeled_in_superclass])
            predictions[test_in_superclass] = classifier.predict(self.test_features[test_in_superclass])

        report_unresolved(unresolved_count, len(self.test_labels))
        return predictions


class MLPEvaluator(Evaluator):

    name = "mlp"

    def fit(self, labeled_indices):
        # A small MLP head (one hidden layer of 256 units) trained on the labeled subset.
        classifier = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=self.seed)
        classifier.fit(self.train_features[labeled_indices], self.train_labels[labeled_indices])
        return classifier

    def predict_within_true_superclass(self, labeled_indices):
        # For the MLP, "superclass known" means: keep only the output classes that live in the
        # test point's true superclass, then take the highest-scoring of those.
        classifier = self.fit(labeled_indices)                              # trained on ALL labeled points
        class_probabilities = classifier.predict_proba(self.test_features)  # (num_test, num_seen_classes)
        seen_classes = classifier.classes_                                  # the fine classes the MLP actually saw
        seen_class_superclasses = self.coarse_map[seen_classes]
        test_superclasses = self.coarse_map[self.test_labels]

        predictions = np.full(len(self.test_labels), -1, dtype=np.int64)
        unresolved_count = 0

        for superclass in np.unique(test_superclasses):
            test_rows = np.where(test_superclasses == superclass)[0]              # test points in this superclass
            allowed_columns = np.where(seen_class_superclasses == superclass)[0]  # output classes in this superclass

            # None of this superclass's fine classes was ever labeled -> nothing allowed to predict.
            if allowed_columns.size == 0:
                unresolved_count += len(test_rows)
                continue

            # Among the allowed classes, pick the highest-probability one for each test point.
            allowed_probabilities = class_probabilities[np.ix_(test_rows, allowed_columns)]
            best_allowed = allowed_probabilities.argmax(axis=1)
            predictions[test_rows] = seen_classes[allowed_columns][best_allowed]

        report_unresolved(unresolved_count, len(self.test_labels))
        return predictions


# name -> class, so --eval can pick a base classifier by string. The superclass oracle
# is a separate flag (--oracle) that composes with whichever base classifier is chosen.
EVALUATORS = {
    OneNearestNeighbourEvaluator.name: OneNearestNeighbourEvaluator,
    MLPEvaluator.name: MLPEvaluator,
}
