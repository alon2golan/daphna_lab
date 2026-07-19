# Flat (non-hierarchical) evaluators -- the family FLOORS and, with --oracle, the CEILINGS.
#   1nn : nearest labeled neighbour over all 100 fine classes.
#   mlp : one MLP head over all 100 fine classes.
# Each also defines its oracle form (restrict to the TRUE superclass) = the family ceiling.

import numpy as np

from .base import Evaluator, make_knn, make_mlp, report_unresolved


class OneNearestNeighbourEvaluator(Evaluator):

    name = "1nn"

    def fit(self, labeled_indices):
        # 1-nearest-neighbour on the labeled subset (features are already L2-normalized).
        classifier = make_knn(n_neighbors=1)
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
            classifier = make_knn(n_neighbors=1)
            classifier.fit(labeled_features[labeled_in_superclass], labeled_fine_labels[labeled_in_superclass])
            predictions[test_in_superclass] = classifier.predict(self.test_features[test_in_superclass])

        report_unresolved(unresolved_count, len(self.test_labels))
        return predictions


class MLPEvaluator(Evaluator):

    name = "mlp"

    def fit(self, labeled_indices):
        # A small MLP head trained on the labeled subset (architecture fixed in base.make_mlp).
        classifier = make_mlp(self.seed)
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
