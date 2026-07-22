# The measurement shared by both curves: train a plain 20-way MLP on a set of embeddings +
# superclass labels, test on the test embeddings, report accuracy. Curve 1 feeds it frozen
# embeddings, curve 2 feeds it finetuned embeddings -- identical protocol, so the only thing
# that differs between the curves is the embedding itself.

import numpy as np
from sklearn.neural_network import MLPClassifier


# --- Probe Configuration Constants ---

MLP_HIDDEN_LAYER_SIZES = (256,)  # one hidden layer of 256, matching the old MLP evaluator
MLP_MAX_ITERATIONS = 300         # epoch cap for the sklearn MLP


# --- Probe ---

def make_mlp(seed):
    """One-hidden-layer MLP probe on the embeddings, seeded for reproducibility."""
    return MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
        max_iter=MLP_MAX_ITERATIONS,
        random_state=seed,
    )


def coarse_mlp_accuracy(probe_train_features, probe_train_coarse, test_features, test_coarse, seed=0):
    """Trains a 20-way MLP on the given (labeled-subset) embeddings and their superclass labels,
    then returns 20-way test accuracy as a percent. Both curves call this identically."""
    probe = make_mlp(seed)                                    # a fresh probe every call
    probe.fit(probe_train_features, probe_train_coarse)       # train on the labeled subset's embeddings
    predictions = probe.predict(test_features)               # predict the held-out test split
    return 100.0 * float((predictions == test_coarse).mean())  # exact-match 20-way accuracy
