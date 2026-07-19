import numpy as np

from .evaluator_base import (HierarchicalEvaluator, make_knn, make_mlp, full_class_probabilities,
                             KNN_SUPERCLASS_NEIGHBOURS)


# --- Nearest-Neighbour Family ---

class HierarchicalKnnEvaluator(HierarchicalEvaluator):
    """
    Two-stage kNN with HARD routing: vote a superclass, then 1-NN inside it.
    The realistic (predicted-superclass) twin of the oracle 1-NN.
    """

    name = "hknn"

    # Neighbour count for the stage-A superclass vote
    superclass_neighbours = KNN_SUPERCLASS_NEIGHBOURS

    def predict(self, labeled_indices):
        """Stage A: kNN vote -> predicted superclass. Stage B: 1-NN within that superclass -> fine class."""
        labeled_features, labeled_fine_labels, labeled_superclasses = self.labeled_arrays(labeled_indices)

        # Stage A: a kNN whose TARGETS are superclasses gives each test point a predicted superclass
        # Never request more neighbours than there are labeled points
        k_superclass = min(self.superclass_neighbours, len(labeled_fine_labels))
        superclass_voter = make_knn(k_superclass)
        superclass_voter.fit(labeled_features, labeled_superclasses)
        predicted_test_superclasses = superclass_voter.predict(self.test_features)

        # Stage B: resolve the fine class one predicted-superclass group at a time
        predictions = np.full(len(self.test_labels), -1, dtype=np.int64)
        for superclass in np.unique(predicted_test_superclasses):
            # Test points routed here, and the labeled candidates that live in this superclass
            test_rows = np.where(predicted_test_superclasses == superclass)[0]
            labeled_in_superclass = labeled_superclasses == superclass

            # The vote only picks a superclass some labeled neighbour has, so this group is never empty
            subclass_classifier = make_knn(n_neighbors=1)
            subclass_classifier.fit(labeled_features[labeled_in_superclass], labeled_fine_labels[labeled_in_superclass])
            predictions[test_rows] = subclass_classifier.predict(self.test_features[test_rows])
        return predictions


class HierarchicalKnnMarginalEvaluator(HierarchicalEvaluator):
    """
    Two-stage kNN with SOFT combination: P(fine) = P(parent | x) * P(fine | parent, x).
    A confident fine vote can survive a 2nd-choice superclass, so coarse errors are recoverable.
    """

    name = "hknn_marginal"

    # Stage-A superclass vote, and stage-B within-superclass fine vote (needs k>1 to yield probabilities)
    superclass_neighbours = KNN_SUPERCLASS_NEIGHBOURS
    subclass_neighbours = KNN_SUPERCLASS_NEIGHBOURS

    def predict(self, labeled_indices):
        """Combine P(superclass|x) with the per-superclass fine vote, then argmax over all fine classes."""
        labeled_features, labeled_fine_labels, labeled_superclasses = self.labeled_arrays(labeled_indices)
        num_test = len(self.test_labels)

        # Stage A: P(superclass | x) for every test point, laid out over all superclass columns
        k_superclass = min(self.superclass_neighbours, len(labeled_fine_labels))
        superclass_voter = make_knn(k_superclass)
        superclass_voter.fit(labeled_features, labeled_superclasses)
        superclass_probabilities = full_class_probabilities(superclass_voter, self.test_features, self.num_superclasses)

        # Accumulate the combined fine score one superclass at a time
        # Each fine class belongs to exactly one superclass, so these column blocks never overlap
        fine_scores = np.zeros((num_test, self.num_classes), dtype=np.float64)
        for superclass in np.unique(labeled_superclasses):
            labeled_in_superclass = labeled_superclasses == superclass

            # Stage B: P(fine | this superclass, x) from a fine kNN trained only inside the superclass
            k_subclass = min(self.subclass_neighbours, int(labeled_in_superclass.sum()))
            fine_voter = make_knn(k_subclass)
            fine_voter.fit(labeled_features[labeled_in_superclass], labeled_fine_labels[labeled_in_superclass])
            conditional_fine_probabilities = full_class_probabilities(fine_voter, self.test_features, self.num_classes)

            # Weight the conditional by P(superclass|x): the factorization P(fine) = P(parent|x) * P(fine|parent,x)
            fine_scores += superclass_probabilities[:, superclass][:, None] * conditional_fine_probabilities
        return fine_scores.argmax(axis=1)


# --- MLP Family ---

class HierarchicalMlpRoutedEvaluator(HierarchicalEvaluator):
    """
    Method #1 -- HARD routing. Stage A is an MLP over superclasses; its argmax routes each test
    point to a dedicated per-superclass fine head. A coarse error is therefore unrecoverable.
    """

    name = "hmlp_routed"

    def predict(self, labeled_indices):
        """Stage A: argmax an MLP superclass head. Stage B: run only that superclass's fine head."""
        labeled_features, labeled_fine_labels, labeled_superclasses = self.labeled_arrays(labeled_indices)

        # Stage A: an MLP superclass head -> a single hard predicted superclass per test point
        superclass_head = make_mlp(self.seed)
        superclass_head.fit(labeled_features, labeled_superclasses)
        predicted_test_superclasses = superclass_head.predict(self.test_features)

        # Stage B: one dedicated fine head per superclass, choosing among only that superclass's classes
        predictions = np.full(len(self.test_labels), -1, dtype=np.int64)
        for superclass in np.unique(predicted_test_superclasses):
            test_rows = np.where(predicted_test_superclasses == superclass)[0]
            labeled_in_superclass = labeled_superclasses == superclass
            fine_labels_here = labeled_fine_labels[labeled_in_superclass]

            # An MLP needs >=2 classes to train; with only one labeled fine class here, it IS the answer
            distinct_fine_classes = np.unique(fine_labels_here)
            if distinct_fine_classes.size == 1:
                predictions[test_rows] = distinct_fine_classes[0]
                continue

            # Otherwise train the per-superclass fine head and predict its routed test points
            fine_head = make_mlp(self.seed)
            fine_head.fit(labeled_features[labeled_in_superclass], fine_labels_here)
            predictions[test_rows] = fine_head.predict(self.test_features[test_rows])
        return predictions


class HierarchicalMlpMarginalEvaluator(HierarchicalEvaluator):
    """
    Method #1 done SOFTLY -- the same heads as hmlp_routed, combined probabilistically:
    score(fine) = P(parent | x) * P(fine | parent, x). A confident fine head can override a
    not-quite-top superclass, so coarse errors are recoverable.
    """

    name = "hmlp_marginal"

    def predict(self, labeled_indices):
        """Combine the MLP superclass probabilities with each per-superclass fine head, then argmax."""
        labeled_features, labeled_fine_labels, labeled_superclasses = self.labeled_arrays(labeled_indices)
        num_test = len(self.test_labels)

        # Stage A: P(superclass | x) over all superclass columns
        superclass_head = make_mlp(self.seed)
        superclass_head.fit(labeled_features, labeled_superclasses)
        superclass_probabilities = full_class_probabilities(superclass_head, self.test_features, self.num_superclasses)

        # Accumulate the combined fine score one superclass at a time
        fine_scores = np.zeros((num_test, self.num_classes), dtype=np.float64)
        for superclass in np.unique(labeled_superclasses):
            labeled_in_superclass = labeled_superclasses == superclass
            fine_labels_here = labeled_fine_labels[labeled_in_superclass]
            distinct_fine_classes = np.unique(fine_labels_here)

            # Stage B: P(fine | this superclass, x). With a single labeled class the conditional is a delta
            if distinct_fine_classes.size == 1:
                conditional_fine_probabilities = np.zeros((num_test, self.num_classes), dtype=np.float64)
                conditional_fine_probabilities[:, distinct_fine_classes[0]] = 1.0
            else:
                fine_head = make_mlp(self.seed)
                fine_head.fit(labeled_features[labeled_in_superclass], fine_labels_here)
                conditional_fine_probabilities = full_class_probabilities(fine_head, self.test_features, self.num_classes)

            # Weight the conditional by the superclass probability and accumulate
            fine_scores += superclass_probabilities[:, superclass][:, None] * conditional_fine_probabilities
        return fine_scores.argmax(axis=1)


class HierarchicalMlpLogitsEvaluator(HierarchicalEvaluator):
    """
    Method #2 -- COARSE-AS-FEATURE. Stage A's soft superclass probabilities are concatenated to
    the embedding and fed to ONE flat fine head. The coarse signal is only a hint (extra input
    dims), so the head can still output any fine class.
    """

    name = "hmlp_logits"

    def predict(self, labeled_indices):
        """Train a superclass head, append its probabilities to the embedding, and run one flat fine head."""
        labeled_features, labeled_fine_labels, labeled_superclasses = self.labeled_arrays(labeled_indices)

        # Stage A: a superclass head, used purely to produce the soft superclass hint vector
        superclass_head = make_mlp(self.seed)
        superclass_head.fit(labeled_features, labeled_superclasses)

        # The hint = P(superclass | x), computed for the labeled points (to train stage B) and the test points
        labeled_superclass_probabilities = full_class_probabilities(superclass_head, labeled_features, self.num_superclasses)
        test_superclass_probabilities = full_class_probabilities(superclass_head, self.test_features, self.num_superclasses)

        # Stage B input = [embedding | superclass hint]; a single flat MLP over all fine classes
        labeled_augmented_features = np.hstack([labeled_features, labeled_superclass_probabilities])
        test_augmented_features = np.hstack([self.test_features, test_superclass_probabilities])
        fine_head = make_mlp(self.seed)
        fine_head.fit(labeled_augmented_features, labeled_fine_labels)
        return fine_head.predict(test_augmented_features)
