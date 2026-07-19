# Hierarchical (two-stage superclass -> subclass) evaluators.
#
# Each PREDICTS the superclass from features (stage A), then the fine class (stage B). It never
# sees the TRUE superclass -- that is the oracle (flat.py), the ceiling these aim at. Two axes:
#   hard route = argmax the superclass, pick the fine class only within it (coarse error is fatal).
#   soft/marginal = score(fine) = P(parent|x) * P(fine|parent,x), argmax over all fine classes
#                   (a confident fine class survives a 2nd-choice superclass -> recovers coarse errors).
# NN family: hknn (hard), hknn_marginal (soft).  MLP family: hmlp_routed (hard), hmlp_marginal
# (soft), hmlp_logits (soft, coarse fed in as extra input features).

import numpy as np

from .base import (HierarchicalEvaluator, make_knn, make_mlp, full_class_probabilities,
                   KNN_SUPERCLASS_NEIGHBOURS)


class HierarchicalKnnEvaluator(HierarchicalEvaluator):
    # Stage A: kNN vote over labeled points -> predicted superclass.
    # Stage B: 1-NN over only the labeled points in that predicted superclass -> fine class.
    # The realistic (predicted-superclass) twin of the oracle 1-NN.

    name = "hknn"
    superclass_neighbours = KNN_SUPERCLASS_NEIGHBOURS   # neighbours for the stage-A superclass vote

    def predict(self, labeled_indices):
        labeled_features, labeled_fine_labels, labeled_superclasses = self.labeled_arrays(labeled_indices)

        # Stage A: a kNN whose targets are superclasses gives each test point a predicted superclass.
        k_superclass = min(self.superclass_neighbours, len(labeled_fine_labels))   # never ask for more neighbours than points
        superclass_voter = make_knn(k_superclass)
        superclass_voter.fit(labeled_features, labeled_superclasses)
        predicted_test_superclasses = superclass_voter.predict(self.test_features)

        # Stage B: inside each predicted superclass, 1-NN over that superclass's labeled points only.
        predictions = np.full(len(self.test_labels), -1, dtype=np.int64)
        for superclass in np.unique(predicted_test_superclasses):
            test_rows = np.where(predicted_test_superclasses == superclass)[0]   # test points routed to this superclass
            labeled_in_superclass = labeled_superclasses == superclass           # its labeled candidates

            # The vote only picks a superclass that some labeled neighbour has, so this is never empty.
            subclass_classifier = make_knn(n_neighbors=1)
            subclass_classifier.fit(labeled_features[labeled_in_superclass], labeled_fine_labels[labeled_in_superclass])
            predictions[test_rows] = subclass_classifier.predict(self.test_features[test_rows])
        return predictions


class HierarchicalKnnMarginalEvaluator(HierarchicalEvaluator):
    # Soft nearest-neighbour version. Stage A gives P(superclass | x); stage B gives, per
    # superclass, P(fine | that superclass, x) from a fine kNN trained inside the superclass.
    # score(fine) = P(parent | x) * P(fine | parent, x), argmaxed over all fine classes.

    name = "hknn_marginal"
    superclass_neighbours = KNN_SUPERCLASS_NEIGHBOURS   # stage-A superclass vote
    subclass_neighbours = KNN_SUPERCLASS_NEIGHBOURS     # stage-B within-superclass fine vote (needs k>1 for probabilities)

    def predict(self, labeled_indices):
        labeled_features, labeled_fine_labels, labeled_superclasses = self.labeled_arrays(labeled_indices)
        num_test = len(self.test_labels)

        # Stage A: P(superclass | x) for every test point, laid out over all superclass columns.
        k_superclass = min(self.superclass_neighbours, len(labeled_fine_labels))
        superclass_voter = make_knn(k_superclass)
        superclass_voter.fit(labeled_features, labeled_superclasses)
        superclass_probabilities = full_class_probabilities(superclass_voter, self.test_features, self.num_superclasses)

        # Accumulate the combined fine score superclass by superclass. Each fine class belongs to
        # exactly one superclass, so these blocks never overlap.
        fine_scores = np.zeros((num_test, self.num_classes), dtype=np.float64)
        for superclass in np.unique(labeled_superclasses):
            labeled_in_superclass = labeled_superclasses == superclass

            # Stage B: P(fine | this superclass, x) from a fine kNN trained only inside the superclass.
            k_subclass = min(self.subclass_neighbours, int(labeled_in_superclass.sum()))
            fine_voter = make_knn(k_subclass)
            fine_voter.fit(labeled_features[labeled_in_superclass], labeled_fine_labels[labeled_in_superclass])
            conditional_fine_probabilities = full_class_probabilities(fine_voter, self.test_features, self.num_classes)

            # Weight by P(superclass | x): the factorization P(fine) = P(parent | x) * P(fine | parent, x).
            fine_scores += superclass_probabilities[:, superclass][:, None] * conditional_fine_probabilities
        return fine_scores.argmax(axis=1)


class HierarchicalMlpRoutedEvaluator(HierarchicalEvaluator):
    # Your method #1: hard routing. Stage A is an MLP over superclasses; its argmax routes each
    # test point to a dedicated per-superclass fine head trained only on that superclass's points.

    name = "hmlp_routed"

    def predict(self, labeled_indices):
        labeled_features, labeled_fine_labels, labeled_superclasses = self.labeled_arrays(labeled_indices)

        # Stage A: MLP superclass head -> a single hard predicted superclass per test point.
        superclass_head = make_mlp(self.seed)
        superclass_head.fit(labeled_features, labeled_superclasses)
        predicted_test_superclasses = superclass_head.predict(self.test_features)

        # Stage B: one fine head per superclass, each choosing among only that superclass's classes.
        predictions = np.full(len(self.test_labels), -1, dtype=np.int64)
        for superclass in np.unique(predicted_test_superclasses):
            test_rows = np.where(predicted_test_superclasses == superclass)[0]
            labeled_in_superclass = labeled_superclasses == superclass
            fine_labels_here = labeled_fine_labels[labeled_in_superclass]

            # An MLP needs >=2 classes to train; if only one fine class was labeled here, it is the answer.
            distinct_fine_classes = np.unique(fine_labels_here)
            if distinct_fine_classes.size == 1:
                predictions[test_rows] = distinct_fine_classes[0]
                continue

            fine_head = make_mlp(self.seed)
            fine_head.fit(labeled_features[labeled_in_superclass], fine_labels_here)
            predictions[test_rows] = fine_head.predict(self.test_features[test_rows])
        return predictions


class HierarchicalMlpMarginalEvaluator(HierarchicalEvaluator):
    # Your method #1 done softly: the same superclass head + per-superclass fine heads as
    # hmlp_routed, but combined probabilistically instead of hard-routed.
    # score(fine) = P(parent | x) * P(fine | parent, x), argmaxed over all fine classes, so a
    # confident fine head can override a not-quite-top superclass.

    name = "hmlp_marginal"

    def predict(self, labeled_indices):
        labeled_features, labeled_fine_labels, labeled_superclasses = self.labeled_arrays(labeled_indices)
        num_test = len(self.test_labels)

        # Stage A: P(superclass | x) over all superclass columns.
        superclass_head = make_mlp(self.seed)
        superclass_head.fit(labeled_features, labeled_superclasses)
        superclass_probabilities = full_class_probabilities(superclass_head, self.test_features, self.num_superclasses)

        fine_scores = np.zeros((num_test, self.num_classes), dtype=np.float64)
        for superclass in np.unique(labeled_superclasses):
            labeled_in_superclass = labeled_superclasses == superclass
            fine_labels_here = labeled_fine_labels[labeled_in_superclass]
            distinct_fine_classes = np.unique(fine_labels_here)

            # Stage B: P(fine | this superclass, x). With a single labeled class the conditional is a delta.
            if distinct_fine_classes.size == 1:
                conditional_fine_probabilities = np.zeros((num_test, self.num_classes), dtype=np.float64)
                conditional_fine_probabilities[:, distinct_fine_classes[0]] = 1.0
            else:
                fine_head = make_mlp(self.seed)
                fine_head.fit(labeled_features[labeled_in_superclass], fine_labels_here)
                conditional_fine_probabilities = full_class_probabilities(fine_head, self.test_features, self.num_classes)

            # Weight the conditional by the superclass probability and accumulate.
            fine_scores += superclass_probabilities[:, superclass][:, None] * conditional_fine_probabilities
        return fine_scores.argmax(axis=1)


class HierarchicalMlpLogitsEvaluator(HierarchicalEvaluator):
    # Your method #2: coarse-as-feature. Stage A is a superclass MLP; its soft probability vector
    # is concatenated to the embedding and fed to ONE flat fine head over all classes. The coarse
    # signal is only a hint (extra input dims), so the head can still pick any fine class.

    name = "hmlp_logits"

    def predict(self, labeled_indices):
        labeled_features, labeled_fine_labels, labeled_superclasses = self.labeled_arrays(labeled_indices)

        # Stage A: superclass head, used purely to produce the soft superclass hint vector.
        superclass_head = make_mlp(self.seed)
        superclass_head.fit(labeled_features, labeled_superclasses)

        # The hint = P(superclass | x); computed for the labeled points (to train stage B) and the test points.
        labeled_superclass_probabilities = full_class_probabilities(superclass_head, labeled_features, self.num_superclasses)
        test_superclass_probabilities = full_class_probabilities(superclass_head, self.test_features, self.num_superclasses)

        # Stage B input = [embedding | superclass hint]; a single flat MLP over all fine classes.
        labeled_augmented_features = np.hstack([labeled_features, labeled_superclass_probabilities])
        test_augmented_features = np.hstack([self.test_features, test_superclass_probabilities])
        fine_head = make_mlp(self.seed)
        fine_head.fit(labeled_augmented_features, labeled_fine_labels)
        return fine_head.predict(test_augmented_features)
