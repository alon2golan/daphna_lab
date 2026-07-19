import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# --- Configuration Constants ---

# Defines the Multi-Layer Perceptron architecture: 1 hidden layer with 256 neurons
MLP_HIDDEN_LAYER_SIZES = (256,)

# Prevents the neural network from training indefinitely by capping epochs at 300
MLP_MAX_ITERATIONS = 300

# Sets the 'k' in k-NN to look at the 10 nearest neighbors for superclass decisions
KNN_SUPERCLASS_NEIGHBOURS = 10


# --- Factory Functions (Model Creators) ---

def make_mlp(seed):
    """Instantiates a standardized MLP classifier to ensure identical neural net setups."""
    return MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
        max_iter=MLP_MAX_ITERATIONS,
        random_state=seed  # Ensures reproducibility across different runs
    )


def make_knn(n_neighbors):
    """Instantiates a standard k-Nearest Neighbors classifier."""
    return KNeighborsClassifier(n_neighbors=n_neighbors)


# --- Helper Utilities ---

def full_class_probabilities(classifier, features, num_columns):
    """
    Transforms scikit-learn's sparse probability matrix into a full-sized matrix 
    where the column index perfectly matches the actual class ID.
    """
    # Get probabilities only for classes the model actually saw during training
    # Shape: (num_samples, num_classes_seen_in_training)
    seen_probabilities = classifier.predict_proba(features)

    # Initialize a blank matrix filled with zeros for ALL possible classes
    # Shape: (num_samples, total_possible_classes_in_dataset)
    full = np.zeros((features.shape[0], num_columns), dtype=np.float64)

    # Place the seen probabilities into their correct, absolute class columns
    # Any class that wasn't seen during training remains at 0.0 probability
    full[:, classifier.classes_] = seen_probabilities
    return full


def report_unresolved(unresolved_count, test_count):
    """Prints a warning if the oracle mode fails due to completely missing training data."""
    if unresolved_count:
        print(
            f"[oracle] {unresolved_count}/{test_count} test points had no labeled example "
            f"in their true superclass -> left unresolved (counted as wrong)",
            flush=True,  # Forces immediate output to the console terminal
        )


# --- Core Evaluator Base Class ---

class Evaluator:
    """Base class for training a classifier on selected data and evaluating test accuracy."""

    name = "base"

    def __init__(self, train_features, train_labels, test_features, test_labels, coarse_map=None, seed=0):
        self.train_features = train_features  # Full pool of training feature vectors
        self.train_labels = train_labels  # Full pool of granular (fine-grained) labels
        self.test_features = test_features  # Features reserved exclusively for testing
        self.test_labels = test_labels  # Granular labels reserved exclusively for testing
        self.coarse_map = coarse_map  # Array mapping fine_label_id -> superclass_id
        self.seed = seed  # Random seed for model consistency

        # Calculate the total number of granular classes (e.g., 100 for CIFAR-100)
        self.num_classes = int(train_labels.max()) + 1

    def fit(self, labeled_indices):
        """Must be overridden by subclasses to train their specific model types."""
        raise NotImplementedError

    def predict(self, labeled_indices):
        """Standard pipeline: fits the model on labeled data, then predicts on test data."""
        classifier = self.fit(labeled_indices)
        return classifier.predict(self.test_features)

    def predict_within_true_superclass(self, labeled_indices):
        """Must be overridden by subclasses to handle Oracle (cheating/ceiling) predictions."""
        raise NotImplementedError

    def _predictions(self, labeled_indices, use_superclass_oracle):
        """Internal router to switch between standard predictions and Oracle predictions."""
        if use_superclass_oracle:
            # Oracle requires the coarse map to know what the 'true' superclasses are
            if self.coarse_map is None:
                raise SystemExit(
                    "--oracle needs a coarse map (coarse_map.npy alongside the embeddings, "
                    "or a built-in dataset like cifar100)"
                )
            return self.predict_within_true_superclass(labeled_indices)

        # Default behavior: normal machine learning prediction
        return self.predict(labeled_indices)

    def accuracy(self, labeled_indices, use_superclass_oracle=False):
        """Calculates what percentage of granular class predictions were exactly right."""
        predictions = self._predictions(labeled_indices, use_superclass_oracle)

        # Returns the mean of a boolean array (True=1, False=0) multiplied by 100 for percentage
        return 100.0 * float((predictions == self.test_labels).mean())

    def fine_and_coarse_accuracy(self, labeled_indices, use_superclass_oracle=False):
        """Computes both granular (fine) accuracy and broad category (coarse) accuracy simultaneously."""
        if self.coarse_map is None:
            raise SystemExit(
                "superclass accuracy needs a coarse map (coarse_map.npy alongside the "
                "embeddings, or a built-in dataset like cifar100)"
            )

        # Generate predictions using whichever mode (standard or oracle) was chosen
        predictions = self._predictions(labeled_indices, use_superclass_oracle)

        # 1. Compute fine accuracy (exact match)
        fine_accuracy = 100.0 * float((predictions == self.test_labels).mean())

        # 2. Compute coarse accuracy (was it at least in the right family category?)
        # Identify which predictions were successfully resolved (oracle uses -1 for failures)
        is_resolved = predictions >= 0

        # Initialize an array of placeholders (-1) for predicted superclasses
        predicted_superclasses = np.full(len(predictions), -1, dtype=np.int64)

        # Map the valid granular predictions up to their parent superclass IDs
        predicted_superclasses[is_resolved] = self.coarse_map[predictions[is_resolved]]

        # Map the actual correct test labels up to their parent superclass IDs
        true_superclasses = self.coarse_map[self.test_labels]

        # Calculate what percentage of predictions landed in the correct parent category
        coarse_accuracy = 100.0 * float((predicted_superclasses == true_superclasses).mean())

        return fine_accuracy, coarse_accuracy


# --- Hierarchical (Two-Stage) Evaluator Mixin ---

class HierarchicalEvaluator(Evaluator):
    """
    Subclass tailored for hierarchical classification models 
    (Stage 1: Predict Superclass -> Stage 2: Predict Fine Subclass).
    """

    name = "hierarchical-base"

    def __init__(self, *args, **kwargs):
        # Initialize the standard Evaluator properties first
        super().__init__(*args, **kwargs)

        # Hierarchical sorting strictly requires a structural map file
        if self.coarse_map is None:
            raise SystemExit(
                f"evaluator '{self.name}' is hierarchical and needs a coarse map "
                "(coarse_map.npy alongside the embeddings, or a built-in dataset like cifar100)"
            )

        # Calculate total parent categories (e.g., 20 superclasses for CIFAR-100)
        self.num_superclasses = int(self.coarse_map.max()) + 1

    def labeled_arrays(self, labeled_indices):
        """Extracts and aligns training subsets using only the currently allowed labeled data indices."""
        # Get the feature vectors for the labeled items
        labeled_features = self.train_features[labeled_indices]

        # Get the granular labels for the labeled items
        labeled_fine_labels = self.train_labels[labeled_indices]

        # Map those fine labels up to determine what superclass each item belongs to
        labeled_superclasses = self.coarse_map[labeled_fine_labels]

        return labeled_features, labeled_fine_labels, labeled_superclasses