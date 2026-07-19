# Evaluation methods -- the overridable "how do we score a labeled set?".
#
# The AL loop hands an Evaluator the labeled indices each round and gets back a
# test accuracy (percent). Swapping --eval changes only HOW the labeled set is
# turned into predictions; the selection loop is untouched. Add a method by
# subclassing Evaluator and registering it in EVALUATORS -- same pattern as the
# selection strategies.

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


class Evaluator:

    name = "base"

    def __init__(self, train_feats, train_labels, test_feats, test_labels, coarse_map=None, seed=0):
        self.train_feats = train_feats   # the pool features (selection happens over these)
        self.train_labels = train_labels
        self.test_feats = test_feats     # held-out features, only ever used to score
        self.test_labels = test_labels
        self.coarse_map = coarse_map     # fine -> superclass map; only some evaluators need it
        self.seed = seed

    def accuracy(self, labeled_idx):
        # percent of test points classified correctly using ONLY the labeled subset
        raise NotImplementedError


# default: 1-nearest-neighbour on the labeled subset (deep_al's NN1_normalized).
class OneNNEvaluator(Evaluator):

    name = "1nn"

    def accuracy(self, labeled_idx):
        clf = KNeighborsClassifier(n_neighbors=1) # features are already L2-normalized
        clf.fit(self.train_feats[labeled_idx], self.train_labels[labeled_idx]) # fit on the labeled subset only
        preds = clf.predict(self.test_feats)
        return 100.0 * float((preds == self.test_labels).mean())


# a small MLP head trained on the labeled subset (needs a few labels per class to help).
class MLPEvaluator(Evaluator):

    name = "mlp"

    def accuracy(self, labeled_idx):
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=self.seed) # one hidden layer
        clf.fit(self.train_feats[labeled_idx], self.train_labels[labeled_idx])
        preds = clf.predict(self.test_feats)
        return 100.0 * float((preds == self.test_labels).mean())


# upper bound "true superclass known for free": each test point's 1-NN searches only
# the labeled points whose class shares its superclass, so it can never miss the coarse
# target -- the gain over plain 1-NN is exactly what a perfect coarse oracle would buy.
class SuperclassOracleEvaluator(Evaluator):

    name = "superclass_oracle"

    def accuracy(self, labeled_idx):
        if self.coarse_map is None:
            raise SystemExit("--eval superclass_oracle needs a coarse map (coarse_map.npy or a built-in dataset)")
        lab_feats = self.train_feats[labeled_idx]
        lab_fine = self.train_labels[labeled_idx]
        lab_super = self.coarse_map[lab_fine]              # superclass of each labeled point
        test_super = self.coarse_map[self.test_labels]     # true superclass of each test point
        preds = np.empty(len(self.test_labels), dtype=lab_fine.dtype)
        for s in np.unique(test_super):
            in_s = test_super == s                          # test points in superclass s
            cand = lab_super == s                           # labeled points allowed as neighbours
            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(lab_feats[cand], lab_fine[cand]) if cand.any() else clf.fit(lab_feats, lab_fine) # fall back to global if empty
            preds[in_s] = clf.predict(self.test_feats[in_s])
        return 100.0 * float((preds == self.test_labels).mean())


# name -> class, so the CLI can pick an evaluation method by string
EVALUATORS = {
    OneNNEvaluator.name: OneNNEvaluator,
    MLPEvaluator.name: MLPEvaluator,
    SuperclassOracleEvaluator.name: SuperclassOracleEvaluator,
}
