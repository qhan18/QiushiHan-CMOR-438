"""Ensemble Methods implemented from scratch."""
import numpy as np
from rice_ml.supervised_learning.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier:
    """Random Forest classifier that combines many decision trees using bagging."""

    def __init__(self, n_estimators=10, max_depth=10,
                 max_features=None, random_state=42):
        self.n_estimators = n_estimators    # how many trees to build
        self.max_depth = max_depth          # max depth for each individual tree
        self.max_features = max_features    # how many features each tree can use
        self.random_state = random_state    # for reproducibility
        self.trees = []                     # will hold all the trained trees
        self.feature_indices = []           # tracks which features each tree used

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        # use sqrt(n_features) by default, which is the standard random forest rule
        max_features = self.max_features or int(np.sqrt(n_features))
        self.trees = []
        self.feature_indices = []

        for _ in range(self.n_estimators):
            # bootstrap: sample training data with replacement so each tree sees different data
            indices = rng.choice(n_samples, n_samples, replace=True)
            # randomly pick a subset of features to further decorrelate the trees
            feature_idx = rng.choice(
                n_features, max_features, replace=False)
            X_sample = X[indices][:, feature_idx]
            y_sample = y[indices]
            # fit a decision tree on this bootstrap sample
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            self.feature_indices.append(feature_idx)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # get predictions from every tree in the forest
        predictions = np.array([
            tree.predict(X[:, feat_idx])
            for tree, feat_idx in zip(self.trees, self.feature_indices)
        ])
        # take a majority vote across all trees for each sample
        return np.array([
            np.bincount(predictions[:, i].astype(int)).argmax()
            for i in range(X.shape[0])
        ])

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))