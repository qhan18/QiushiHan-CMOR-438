"""Decision Tree Classifier implemented from scratch."""
import numpy as np


class Node:
    """Represents a node in the decision tree."""
    def __init__(self, feature=None, threshold=None, left=None,
                 right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier:
    """Decision tree classifier using information gain."""

    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.root = self._grow_tree(X, y)
        return self

    def _entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-10))

    def _information_gain(self, y, left_y, right_y):
        n = len(y)
        return (self._entropy(y) -
                (len(left_y) / n) * self._entropy(left_y) -
                (len(right_y) / n) * self._entropy(right_y))

    def _best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                gain = self._information_gain(
                    y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):
        if (depth >= self.max_depth or
                len(y) < self.min_samples_split or
                len(np.unique(y)) == 1):
            value, counts = np.unique(y, return_counts=True)
            return Node(value=value[np.argmax(counts)])
        feature, threshold = self._best_split(X, y)
        if feature is None:
            value, counts = np.unique(y, return_counts=True)
            return Node(value=value[np.argmax(counts)])
        left_mask = X[:, feature] <= threshold
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature=feature, threshold=threshold,
                    left=left, right=right)

    def _predict_single(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_single(x, self.root) for x in X])

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))