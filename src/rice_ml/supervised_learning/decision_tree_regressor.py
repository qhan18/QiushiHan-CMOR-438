"""Decision Tree Regressor implemented from scratch."""
import numpy as np


class NodeR:
    """A single node in the regression tree, either a split or a leaf."""
    def __init__(self, feature=None, threshold=None, left=None,
                 right=None, value=None):
        self.feature = feature      # which feature we're splitting on
        self.threshold = threshold  # the cutoff value for the split
        self.left = left            # subtree for samples where feature <= threshold
        self.right = right          # subtree for samples where feature > threshold
        self.value = value          # only set if this is a leaf node

    def is_leaf(self):
        # leaf nodes have a value, internal nodes dont
        return self.value is not None


class DecisionTreeRegressor:
    """Decision tree regressor that splits on variance reduction."""

    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth                  # how deep the tree can grow
        self.min_samples_split = min_samples_split  # dont split if too few samples
        self.root = None                            # tree gets built during fit

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # start growing the tree recursively from the top
        self.root = self._grow_tree(X, y)
        return self

    def _variance_reduction(self, y, left_y, right_y):
        # how much does this split reduce the variance in the targets
        n = len(y)
        parent_var = np.var(y)
        left_var = (len(left_y) / n) * np.var(left_y)
        right_var = (len(right_y) / n) * np.var(right_y)
        return parent_var - left_var - right_var

    def _best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None
        # brute force search over all features and thresholds
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                # skip if the split doesnt actually divide the data
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                gain = self._variance_reduction(
                    y, y[left_mask], y[right_mask])
                # save this split if its the best one so far
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):
        # base cases: tree is too deep or not enough samples to split
        if (depth >= self.max_depth or
                len(y) < self.min_samples_split):
            # leaf predicts the average of all remaining target values
            return NodeR(value=np.mean(y))
        feature, threshold = self._best_split(X, y)
        # if no good split found just return a leaf
        if feature is None:
            return NodeR(value=np.mean(y))
        # split the data and keep growing each side
        left_mask = X[:, feature] <= threshold
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)
        return NodeR(feature=feature, threshold=threshold,
                     left=left, right=right)

    def _predict_single(self, x, node):
        # walk down the tree until we hit a leaf
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_single(x, self.root) for x in X])

    def score(self, X, y):
        """R squared, measures how well predictions match actual values."""
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        # residual sum of squares
        ss_res = np.sum((y - y_pred) ** 2)
        # total sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot