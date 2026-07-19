# Framework for evaluation methods to score a labeled subset.

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


class Evaluator:
    name = "base"

    def __init__(self, train_feats, train_labels, test_feats, test_labels, coarse_map=None, seed=0):
        self.train_feats = train_feats  # Store pool features
        self.train_labels = train_labels  # Store pool labels
        self.test_feats = test_feats  # Store test features
        self.test_labels = test_labels  # Store test labels
        self.coarse_map = coarse_map  # Store superclass map
        self.seed = seed  # Store random seed

    def accuracy(self, labeled_idx):
        """Computes test set accuracy (percent) using only the labeled subset."""
        raise NotImplementedError  # Force subclass override


class OneNNEvaluator(Evaluator):
    name = "1nn"

    def accuracy(self, labeled_idx):
        clf = KNeighborsClassifier(n_neighbors=1)  # Init 1-NN model
        clf.fit(self.train_feats[labeled_idx], self.train_labels[labeled_idx])  # Train on labeled pool
        preds = clf.predict(self.test_feats)  # Predict test set
        return 100.0 * float((preds == self.test_labels).mean())  # Calculate accuracy %


class MLPEvaluator(Evaluator):
    name = "mlp"

    def accuracy(self, labeled_idx):
        # Init MLP network with 1 hidden layer of 256 units
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=self.seed)
        clf.fit(self.train_feats[labeled_idx], self.train_labels[labeled_idx])  # Train on labeled pool
        preds = clf.predict(self.test_feats)  # Predict test set
        return 100.0 * float((preds == self.test_labels).mean())  # Calculate accuracy %


class SuperclassOracleEvaluator(Evaluator):
    name = "superclass_oracle"
    """
        Computes an upper-bound accuracy score by giving the model a "coarse oracle" helper.
        When predicting a test point's specific fine label, the 1-NN search space is strictly 
        restricted to training samples that share the exact same broad superclass category. 
        This measures performance gains achieved when broad class confusion is eliminated.
        """

    def accuracy(self, labeled_idx):
        if self.coarse_map is None:
            raise SystemExit("Error: --eval superclass_oracle requires a valid coarse map matrix.")

        labeled_features = self.train_feats[labeled_idx]  # Get labeled pool features
        labeled_fine_labels = self.train_labels[labeled_idx]  # Get labeled pool fine labels

        labeled_superclasses = self.coarse_map[labeled_fine_labels]  # Get labeled pool superclasses
        test_superclasses = self.coarse_map[self.test_labels]  # Get test set superclasses
        # -1 = unresolved: never equals a real label, so it counts as a miss. int64 avoids uint wrap-around.
        test_predictions = np.full(len(self.test_labels), -1, dtype=np.int64)
        unresolved = 0  # test points whose true superclass has no labeled exemplar yet

        for superclass_id in np.unique(test_superclasses):  # Loop through unique superclasses
            test_points_in_superclass = test_superclasses == superclass_id  # Mask matching test points
            candidate_pool_in_superclass = labeled_superclasses == superclass_id  # Mask matching training points

            # Nothing labeled in this true superclass yet -> there is genuinely nothing to search.
            # Leave these points unresolved (a miss); do NOT answer them from another superclass,
            # which would violate the "true superclass known" premise this evaluator measures.
            if not candidate_pool_in_superclass.any():
                unresolved += int(test_points_in_superclass.sum())
                continue

            classifier = KNeighborsClassifier(n_neighbors=1)  # 1-NN restricted to the known superclass
            classifier.fit(labeled_features[candidate_pool_in_superclass],
                           labeled_fine_labels[candidate_pool_in_superclass])
            test_predictions[test_points_in_superclass] = classifier.predict(self.test_feats[test_points_in_superclass])

        # Make that decision visible instead of hiding it inside the accuracy number
        if unresolved:
            print(f"[superclass_oracle] {unresolved}/{len(self.test_labels)} test points had no labeled "
                  f"exemplar in their superclass -> left unresolved (counted as wrong)", flush=True)

        return 100.0 * float((test_predictions == self.test_labels).mean())  # Calculate accuracy %


# Registry dictionary for dynamic CLI string lookup mapping
EVALUATORS = {
    OneNNEvaluator.name: OneNNEvaluator,
    MLPEvaluator.name: MLPEvaluator,
    SuperclassOracleEvaluator.name: SuperclassOracleEvaluator,
}