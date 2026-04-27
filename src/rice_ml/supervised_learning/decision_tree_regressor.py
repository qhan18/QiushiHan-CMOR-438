"""Decision Tree Regressor implemented from scratch."""
import numpy as np


class NodeR:
    """Represents a node in the regression tree."""
    def __init__(self, feature=None, threshold=None, left=None,
                 right=None, value=None):
        self.feature = feature      # feature index to split on
        self.threshold = threshold  # threshold value for the split
        self.left = left            # left subtree (feature <= threshold)
        self.right = right          # right subtree (feature > threshold)
        self.value = value          # predicted mean value (leaf nodes only)

    def is_leaf(self):
        return self.value is not None


class DecisionTreeRegressor:
    """Decision tree regressor using variance reduction."""

    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth                  # maximum tree depth
        self.min_samples_split = min_samples_split  # minimum samples to split
        self.root = None                            # root node of the tree

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # recursively grow the tree from the root
        self.root = self._grow_tree(X, y)
        return self

    def _variance_reduction(self, y, left_y, right_y):
        # compute reduction in variance after a split
        n = len(y)
        return (np.var(y) -
                (len(left_y) / n) * np.var(left_y) -
                (len(right_y) / n) * np.var(right_y))

    def _best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None
        # try every feature and every unique threshold value
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                gain = self._variance_reduction(
                    y, y[left_mask], y[right_mask])
                # keep the split with the highest variance reduction
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):
        # stopping conditions: max depth or too few samples
        if (depth >= self.max_depth or
                len(y) < self.min_samples_split):
            # leaf node predicts the mean of remaining samples
            return NodeR(value=np.mean(y))
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return NodeR(value=np.mean(y))
        # split data and recursively grow left and right subtrees
        left_mask = X[:, feature] <= threshold
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)
        return NodeR(feature=feature, threshold=threshold,
                     left=left, right=right)

    def _predict_single(self, x, node):
        # traverse the tree until reaching a leaf node
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_single(x, self.root) for x in X])

    def score(self, X, y):
        """R² coefficient of determination."""
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot