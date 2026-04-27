"""Ensemble Methods implemented from scratch."""
import numpy as np
from rice_ml.supervised_learning.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier:
    """Random Forest using bagging with decision trees."""

    def __init__(self, n_estimators=10, max_depth=10,
                 max_features=None, random_state=42):
        self.n_estimators = n_estimators    # number of trees in the forest
        self.max_depth = max_depth          # maximum depth of each tree
        self.max_features = max_features    # number of features per split
        self.random_state = random_state    # seed for reproducibility
        self.trees = []                     # list of trained decision trees
        self.feature_indices = []           # feature subsets used per tree

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        # default feature count is sqrt(n_features) as per random forest convention
        max_features = self.max_features or int(np.sqrt(n_features))
        self.trees = []
        self.feature_indices = []

        for _ in range(self.n_estimators):
            # bootstrap sample: sample with replacement
            indices = rng.choice(n_samples, n_samples, replace=True)
            # randomly select a subset of features for this tree
            feature_idx = rng.choice(
                n_features, max_features, replace=False)
            X_sample = X[indices][:, feature_idx]
            y_sample = y[indices]
            # train a decision tree on this bootstrap sample
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            self.feature_indices.append(feature_idx)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # collect predictions from all trees
        predictions = np.array([
            tree.predict(X[:, feat_idx])
            for tree, feat_idx in zip(self.trees, self.feature_indices)
        ])
        # majority vote across all trees for each sample
        return np.array([
            np.bincount(predictions[:, i].astype(int)).argmax()
            for i in range(X.shape[0])
        ])

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))