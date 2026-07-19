import numpy as np

from .evaluator_base import (HierarchicalEvaluator, make_knn, make_mlp, full_class_probabilities,
                             report_unresolved, KNN_SUPERCLASS_NEIGHBOURS)


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

        # Unpack allowed training arrays: features, granular labels, parent category IDs
        labeled_features, labeled_fine_labels, labeled_superclasses = self.labeled_arrays(labeled_indices)

        # Ensure we don't look for more neighbors than total training items available
        k_superclass = min(self.superclass_neighbours, len(labeled_fine_labels))

        # Instantiate a kNN model dedicated to predicting parent categories
        superclass_voter = make_knn(k_superclass)

        # Train on the labeled subset using broad superclasses as targets
        superclass_voter.fit(labeled_features, labeled_superclasses)

        # Generate a hard parent-category prediction for every single test point
        predicted_test_superclasses = superclass_voter.predict(self.test_features)

        # Initialize an array of placeholders (-1) to store our final granular predictions
        predictions = np.full(len(self.test_labels), -1, dtype=np.int64)

        # Loop through each unique parent category that was predicted at least once
        for superclass in np.unique(predicted_test_superclasses):
            # Locate all test data rows routed to this specific parent category
            test_rows = np.where(predicted_test_superclasses == superclass)[0]

            # Locate all training data items belonging to this specific parent category
            labeled_in_superclass = labeled_superclasses == superclass

            # Create a localized 1-NN model to handle the final granular choice
            subclass_classifier = make_knn(n_neighbors=1)

            # Train using ONLY the training items that belong to this superclass
            subclass_classifier.fit(labeled_features[labeled_in_superclass], labeled_fine_labels[labeled_in_superclass])

            # Predict granular labels for the routed test points and store them
            predictions[test_rows] = subclass_classifier.predict(self.test_features[test_rows])

        return predictions

    def predict_within_true_superclass(self, labeled_indices):
        """Matched ceiling: route by the TRUE superclass, then 1-NN inside it (equivalent to oracle-1nn)."""

        # Unpack training arrays and read the TRUE superclass of every test point
        labeled_features, labeled_fine_labels, labeled_superclasses = self.labeled_arrays(labeled_indices)
        test_superclasses = self.coarse_map[self.test_labels]

        # Start unresolved; resolve the test points one TRUE-superclass group at a time
        predictions = np.full(len(self.test_labels), -1, dtype=np.int64)
        unresolved_count = 0
        for superclass in np.unique(test_superclasses):
            test_in_superclass = test_superclasses == superclass
            labeled_in_superclass = labeled_superclasses == superclass

            # No labeled example in this true superclass -> nothing to search, leave unresolved
            if not labeled_in_superclass.any():
                unresolved_count += int(test_in_superclass.sum())
                continue

            # 1-NN restricted to the labeled points of the TRUE superclass
            subclass_classifier = make_knn(n_neighbors=1)
            subclass_classifier.fit(labeled_features[labeled_in_superclass], labeled_fine_labels[labeled_in_superclass])
            predictions[test_in_superclass] = subclass_classifier.predict(self.test_features[test_in_superclass])

        report_unresolved(unresolved_count, len(self.test_labels))
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

        # Extract the current active training data arrays
        labeled_features, labeled_fine_labels, labeled_superclasses = self.labeled_arrays(labeled_indices)
        num_test = len(self.test_labels)

        # Cap neighbors to avoid requesting more than the total training samples
        k_superclass = min(self.superclass_neighbours, len(labeled_fine_labels))

        # Set up and train the broad parent-category voter
        superclass_voter = make_knn(k_superclass)
        superclass_voter.fit(labeled_features, labeled_superclasses)

        # Calculate soft parent-category probability distributions across all test points
        superclass_probabilities = full_class_probabilities(superclass_voter, self.test_features, self.num_superclasses)

        # Initialize a zero-filled score grid for all test points across all granular classes
        fine_scores = np.zeros((num_test, self.num_classes), dtype=np.float64)

        # Loop over every unique parent category present in the current training set
        for superclass in np.unique(labeled_superclasses):
            # Mask tracking which training samples belong to this parent category
            labeled_in_superclass = labeled_superclasses == superclass

            # Cap local neighbors based on available training items in this group
            k_subclass = min(self.subclass_neighbours, int(labeled_in_superclass.sum()))

            # Set up and train a granular subclass voter restricted to this family group
            fine_voter = make_knn(k_subclass)
            fine_voter.fit(labeled_features[labeled_in_superclass], labeled_fine_labels[labeled_in_superclass])

            # Get fine-grained probabilities for all test items within this category block
            conditional_fine_probabilities = full_class_probabilities(fine_voter, self.test_features, self.num_classes)

            # Weight the sub-probabilities by the parent probability and accumulate them
            fine_scores += superclass_probabilities[:, superclass][:, None] * conditional_fine_probabilities

        # Select the absolute highest scoring granular class ID for each test point
        return fine_scores.argmax(axis=1)

    def predict_within_true_superclass(self, labeled_indices):
        """Matched ceiling: the true superclass collapses the marginal to a fine kNN vote inside it."""

        # Unpack training arrays and read the TRUE superclass of every test point
        labeled_features, labeled_fine_labels, labeled_superclasses = self.labeled_arrays(labeled_indices)
        test_superclasses = self.coarse_map[self.test_labels]

        # Start unresolved; resolve the test points one TRUE-superclass group at a time
        predictions = np.full(len(self.test_labels), -1, dtype=np.int64)
        unresolved_count = 0
        for superclass in np.unique(test_superclasses):
            test_in_superclass = test_superclasses == superclass
            labeled_in_superclass = labeled_superclasses == superclass

            # No labeled example in this true superclass -> leave unresolved
            if not labeled_in_superclass.any():
                unresolved_count += int(test_in_superclass.sum())
                continue

            # Fine kNN vote restricted to the TRUE superclass's labeled points
            k_subclass = min(self.subclass_neighbours, int(labeled_in_superclass.sum()))
            fine_voter = make_knn(k_subclass)
            fine_voter.fit(labeled_features[labeled_in_superclass], labeled_fine_labels[labeled_in_superclass])
            predictions[test_in_superclass] = fine_voter.predict(self.test_features[test_in_superclass])

        report_unresolved(unresolved_count, len(self.test_labels))
        return predictions


# --- MLP Family ---

class HierarchicalMlpRoutedEvaluator(HierarchicalEvaluator):
    """
    Method #1 -- HARD routing. Stage A is an MLP over superclasses; its argmax routes each test
    point to a dedicated per-superclass fine head. A coarse error is therefore unrecoverable.
    """

    name = "hmlp_routed"

    def predict(self, labeled_indices):
        """Stage A: argmax an MLP superclass head. Stage B: run only that superclass's fine head."""
        # Unpack training vectors and category layouts for the current cycle
        labeled_features, labeled_fine_labels, labeled_superclasses = self.labeled_arrays(labeled_indices)

        # Build and train an MLP dedicated solely to recognizing parent categories
        superclass_head = make_mlp(self.seed)
        superclass_head.fit(labeled_features, labeled_superclasses)

        # Generate a hard parent category prediction for each test vector
        predicted_test_superclasses = superclass_head.predict(self.test_features)

        # Set up a blank array to hold final granular classification results
        predictions = np.full(len(self.test_labels), -1, dtype=np.int64)

        # Loop through each parent category predicted by the top-level network
        for superclass in np.unique(predicted_test_superclasses):
            # Isolate rows of test elements that were routed down this branch
            test_rows = np.where(predicted_test_superclasses == superclass)[0]

            # Find training examples belonging strictly to this parent category
            labeled_in_superclass = labeled_superclasses == superclass
            fine_labels_here = labeled_fine_labels[labeled_in_superclass]

            # If only 1 fine class exists in training, it's the only possible answer
            distinct_fine_classes = np.unique(fine_labels_here)
            if distinct_fine_classes.size == 1:
                predictions[test_rows] = distinct_fine_classes[0]
                continue

            # Otherwise, train a dedicated sub-MLP for this specific family group
            fine_head = make_mlp(self.seed)
            fine_head.fit(labeled_features[labeled_in_superclass], fine_labels_here)

            # Evaluate the routed test rows using this specialized neural head
            predictions[test_rows] = fine_head.predict(self.test_features[test_rows])

        return predictions

    def predict_within_true_superclass(self, labeled_indices):
        """Matched ceiling: route by the TRUE superclass to its dedicated per-superclass fine head."""

        # Unpack training arrays and read the TRUE superclass of every test point
        labeled_features, labeled_fine_labels, labeled_superclasses = self.labeled_arrays(labeled_indices)
        test_superclasses = self.coarse_map[self.test_labels]

        # Start unresolved; run each TRUE-superclass group's dedicated head
        predictions = np.full(len(self.test_labels), -1, dtype=np.int64)
        unresolved_count = 0
        for superclass in np.unique(test_superclasses):
            test_rows = np.where(test_superclasses == superclass)[0]
            labeled_in_superclass = labeled_superclasses == superclass

            # No labeled example in this true superclass -> no head to run, leave unresolved
            if not labeled_in_superclass.any():
                unresolved_count += len(test_rows)
                continue

            fine_labels_here = labeled_fine_labels[labeled_in_superclass]
            distinct_fine_classes = np.unique(fine_labels_here)

            # A single labeled fine class is the only possible answer
            if distinct_fine_classes.size == 1:
                predictions[test_rows] = distinct_fine_classes[0]
                continue

            # The dedicated per-superclass fine head, now given the TRUE superclass
            fine_head = make_mlp(self.seed)
            fine_head.fit(labeled_features[labeled_in_superclass], fine_labels_here)
            predictions[test_rows] = fine_head.predict(self.test_features[test_rows])

        report_unresolved(unresolved_count, len(self.test_labels))
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
        # Extract active training inputs and category references
        labeled_features, labeled_fine_labels, labeled_superclasses = self.labeled_arrays(labeled_indices)
        num_test = len(self.test_labels)

        # Build and train the master parent-category neural head
        superclass_head = make_mlp(self.seed)
        superclass_head.fit(labeled_features, labeled_superclasses)

        # Compute soft parent-category probability distributions across all test vectors
        superclass_probabilities = full_class_probabilities(superclass_head, self.test_features, self.num_superclasses)

        # Initialize an empty accumulator score table for all granular classes
        fine_scores = np.zeros((num_test, self.num_classes), dtype=np.float64)

        # Iterate over every unique parent category known in the current training set
        for superclass in np.unique(labeled_superclasses):
            # Target all training samples belonging to this parent category
            labeled_in_superclass = labeled_superclasses == superclass
            fine_labels_here = labeled_fine_labels[labeled_in_superclass]
            distinct_fine_classes = np.unique(fine_labels_here)

            # If only 1 fine class exists, its local probability is a absolute certainty (1.0)
            if distinct_fine_classes.size == 1:
                conditional_fine_probabilities = np.zeros((num_test, self.num_classes), dtype=np.float64)
                conditional_fine_probabilities[:, distinct_fine_classes[0]] = 1.0
            else:
                # Build and train a specialized sub-MLP head for this parent category
                fine_head = make_mlp(self.seed)
                fine_head.fit(labeled_features[labeled_in_superclass], fine_labels_here)

                # Extract its fine-grained probability outputs across the full label space
                conditional_fine_probabilities = full_class_probabilities(fine_head, self.test_features,
                                                                          self.num_classes)

            # Scale the subclass probabilities by the parent network's confidence and add to totals
            fine_scores += superclass_probabilities[:, superclass][:, None] * conditional_fine_probabilities

        # Pinpoint and return the index holding the highest overall calculated probability score
        return fine_scores.argmax(axis=1)

    def predict_within_true_superclass(self, labeled_indices):
        """Matched ceiling: the true one-hot superclass reduces the marginal to its per-superclass fine head."""

        # Unpack training arrays and read the TRUE superclass of every test point
        labeled_features, labeled_fine_labels, labeled_superclasses = self.labeled_arrays(labeled_indices)
        test_superclasses = self.coarse_map[self.test_labels]

        # Start unresolved; run each TRUE-superclass group's dedicated fine head
        predictions = np.full(len(self.test_labels), -1, dtype=np.int64)
        unresolved_count = 0
        for superclass in np.unique(test_superclasses):
            test_rows = np.where(test_superclasses == superclass)[0]
            labeled_in_superclass = labeled_superclasses == superclass

            # No labeled example in this true superclass -> leave unresolved
            if not labeled_in_superclass.any():
                unresolved_count += len(test_rows)
                continue

            fine_labels_here = labeled_fine_labels[labeled_in_superclass]
            distinct_fine_classes = np.unique(fine_labels_here)

            # A single labeled fine class is the only possible answer
            if distinct_fine_classes.size == 1:
                predictions[test_rows] = distinct_fine_classes[0]
                continue

            # The dedicated per-superclass fine head, argmax within the TRUE superclass
            fine_head = make_mlp(self.seed)
            fine_head.fit(labeled_features[labeled_in_superclass], fine_labels_here)
            predictions[test_rows] = fine_head.predict(self.test_features[test_rows])

        report_unresolved(unresolved_count, len(self.test_labels))
        return predictions


class HierarchicalMlpLogitsEvaluator(HierarchicalEvaluator):
    """
    Method #2 -- COARSE-AS-FEATURE. Stage A's soft superclass probabilities are concatenated to
    the embedding and fed to ONE flat fine head. The coarse signal is only a hint (extra input
    dims), so the head can still output any fine class.
    """

    name = "hmlp_logits"

    def predict(self, labeled_indices):
        """Train a superclass head, append its probabilities to the embedding, and run one flat fine head."""
        # Unpack valid training feature matrices and structural target indicators
        labeled_features, labeled_fine_labels, labeled_superclasses = self.labeled_arrays(labeled_indices)

        # Construct and fit the master parent category neural network
        superclass_head = make_mlp(self.seed)
        superclass_head.fit(labeled_features, labeled_superclasses)

        # Calculate parent category hint distributions for the training features
        labeled_superclass_probabilities = full_class_probabilities(superclass_head, labeled_features,
                                                                    self.num_superclasses)

        # Calculate matching parent category hint distributions for the test features
        test_superclass_probabilities = full_class_probabilities(superclass_head, self.test_features,
                                                                 self.num_superclasses)

        # Stitch original vectors together with parent category probabilities side-by-side
        labeled_augmented_features = np.hstack([labeled_features, labeled_superclass_probabilities])
        test_augmented_features = np.hstack([self.test_features, test_superclass_probabilities])

        # Build one giant flat network to handle the final granular choice
        fine_head = make_mlp(self.seed)

        # Train this flat network using the enriched, category-aware feature vectors
        fine_head.fit(labeled_augmented_features, labeled_fine_labels)

        # Return standard predictions generated by processing the enriched test vectors
        return fine_head.predict(test_augmented_features)

    def predict_within_true_superclass(self, labeled_indices):
        """Matched ceiling: feed the TRUE one-hot superclass as the hint instead of the predicted probabilities."""

        # Unpack training arrays and read the TRUE superclass of every test point
        labeled_features, labeled_fine_labels, labeled_superclasses = self.labeled_arrays(labeled_indices)
        test_superclasses = self.coarse_map[self.test_labels]

        # One-hot the TRUE superclass for labeled and test points -> a perfect coarse hint
        labeled_hint = np.zeros((len(labeled_fine_labels), self.num_superclasses), dtype=np.float64)
        labeled_hint[np.arange(len(labeled_fine_labels)), labeled_superclasses] = 1.0
        test_hint = np.zeros((len(self.test_labels), self.num_superclasses), dtype=np.float64)
        test_hint[np.arange(len(self.test_labels)), test_superclasses] = 1.0

        # Same augmented flat head as predict(), but trained/tested with the true one-hot appended
        labeled_augmented_features = np.hstack([labeled_features, labeled_hint])
        test_augmented_features = np.hstack([self.test_features, test_hint])
        fine_head = make_mlp(self.seed)
        fine_head.fit(labeled_augmented_features, labeled_fine_labels)

        # No output restriction: the coarse info is a perfect input feature, not a hard filter
        return fine_head.predict(test_augmented_features)