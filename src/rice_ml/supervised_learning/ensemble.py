"""Ensemble Methods implemented from scratch."""
import numpy as np
from rice_ml.supervised_learning.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier:
    """Random Forest using bagging with decision trees."""

    def __init__(self, n_estimators=10, max_depth=10,
                 max_features=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        max_features = self.max_features or int(np.sqrt(n_features))
        self.trees = []
        self.feature_indices = []

        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = rng.choice(n_samples, n_samples, replace=True)
            feature_idx = rng.choice(
                n_features, max_features, replace=False)
            X_sample = X[indices][:, feature_idx]
            y_sample = y[indices]
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            self.feature_indices.append(feature_idx)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        predictions = np.array([
            tree.predict(X[:, feat_idx])
            for tree, feat_idx in zip(self.trees, self.feature_indices)
        ])
        return np.array([
            np.bincount(predictions[:, i].astype(int)).argmax()
            for i in range(X.shape[0])
        ])

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))