import numpy as np

from .evaluator_base import Evaluator, make_knn, make_mlp, report_unresolved


# --- 1-Nearest-Neighbour Evaluator ---

class OneNearestNeighbourEvaluator(Evaluator):
    """Flat 1-NN over all fine classes: the nearest-neighbour family FLOOR (and, with --oracle, its CEILING)."""

    name = "1nn"

    def fit(self, labeled_indices):
        """Fits a 1-NN on the labeled subset (features are already L2-normalized, so Euclidean = cosine order)."""
        classifier = make_knn(n_neighbors=1)
        classifier.fit(self.train_features[labeled_indices], self.train_labels[labeled_indices])
        return classifier

    def predict_within_true_superclass(self, labeled_indices):
        """Oracle 1-NN: for each test point, search ONLY the labeled points inside its true superclass."""

        # Pull the labeled subset and roll its fine labels up to superclasses
        labeled_features = self.train_features[labeled_indices]
        labeled_fine_labels = self.train_labels[labeled_indices]
        labeled_superclasses = self.coarse_map[labeled_fine_labels]

        # The true superclass of every test point (this is the information the oracle is handed)
        test_superclasses = self.coarse_map[self.test_labels]

        # Start every prediction at -1 (unresolved); -1 never equals a real label, so it counts as a miss
        predictions = np.full(len(self.test_labels), -1, dtype=np.int64)
        unresolved_count = 0

        # Answer the test points one superclass at a time
        for superclass in np.unique(test_superclasses):
            # The test points to answer this pass, and the labeled candidates allowed to answer them
            test_in_superclass = test_superclasses == superclass
            labeled_in_superclass = labeled_superclasses == superclass

            # No labeled example in this true superclass -> genuinely nothing to search, leave unresolved
            if not labeled_in_superclass.any():
                unresolved_count += int(test_in_superclass.sum())
                continue

            # 1-NN restricted to the same-superclass labeled points
            classifier = make_knn(n_neighbors=1)
            classifier.fit(labeled_features[labeled_in_superclass], labeled_fine_labels[labeled_in_superclass])
            predictions[test_in_superclass] = classifier.predict(self.test_features[test_in_superclass])

        # Announce how many points had no candidate at all (counted as wrong)
        report_unresolved(unresolved_count, len(self.test_labels))
        return predictions


# --- MLP Evaluator ---

class MLPEvaluator(Evaluator):
    """Flat MLP over all fine classes: the MLP family FLOOR (and, with --oracle, its CEILING)."""

    name = "mlp"

    def fit(self, labeled_indices):
        """Fits a single MLP head on the labeled subset (architecture fixed in evaluator_base.make_mlp)."""
        classifier = make_mlp(self.seed)
        classifier.fit(self.train_features[labeled_indices], self.train_labels[labeled_indices])
        return classifier

    def predict_within_true_superclass(self, labeled_indices):
        """Oracle MLP: keep only the output classes living in each test point's true superclass, then argmax."""

        # Train one flat head on ALL labeled points, then read its per-class probabilities
        classifier = self.fit(labeled_indices)
        # Shape: (num_test, num_classes_seen_in_training)
        class_probabilities = classifier.predict_proba(self.test_features)

        # The fine classes the MLP actually saw, and the superclass each of them rolls up to
        seen_classes = classifier.classes_
        seen_class_superclasses = self.coarse_map[seen_classes]

        # The true superclass of every test point (the oracle's free information)
        test_superclasses = self.coarse_map[self.test_labels]

        # Start every prediction at -1 (unresolved)
        predictions = np.full(len(self.test_labels), -1, dtype=np.int64)
        unresolved_count = 0

        # Resolve the test points one superclass at a time
        for superclass in np.unique(test_superclasses):
            # Rows = test points in this superclass; columns = output classes belonging to it
            test_rows = np.where(test_superclasses == superclass)[0]
            allowed_columns = np.where(seen_class_superclasses == superclass)[0]

            # None of this superclass's fine classes was ever labeled -> nothing allowed to predict
            if allowed_columns.size == 0:
                unresolved_count += len(test_rows)
                continue

            # 1. Slice out only the allowed rows and columns
            allowed_probabilities = class_probabilities[np.ix_(test_rows, allowed_columns)]

            # 2. Find the index of the highest score per row
            best_allowed = allowed_probabilities.argmax(axis=1)

            # 3. Map relative index to real Class ID and save
            predictions[test_rows] = seen_classes[allowed_columns][best_allowed]

        # Announce how many points had no allowed class at all (counted as wrong)
        report_unresolved(unresolved_count, len(self.test_labels))
        return predictions
