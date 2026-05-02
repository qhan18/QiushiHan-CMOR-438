"""Decision Tree Classifier implemented from scratch."""
import numpy as np


class Node:
    """
    A single node in the decision tree.

    Can be either an internal node that splits on a feature threshold
    or a leaf node that returns a predicted class label.

    Parameters
    ----------
    feature : int or None
        Index of the feature to split on. None for leaf nodes.
    threshold : float or None
        Threshold value for the split. None for leaf nodes.
    left : Node or None
        Left subtree for samples where feature <= threshold.
    right : Node or None
        Right subtree for samples where feature > threshold.
    value : int or None
        Predicted class label. Only set for leaf nodes.
    """

    def __init__(self, feature=None, threshold=None, left=None,
                 right=None, value=None):
        self.feature = feature      # which feature we're splitting on
        self.threshold = threshold  # the cutoff value for the split
        self.left = left            # subtree for samples where feature <= threshold
        self.right = right          # subtree for samples where feature > threshold
        self.value = value          # only set if this is a leaf node

    def is_leaf(self):
        """Return True if this node is a leaf node."""
        # leaf nodes have a value, internal nodes dont
        return self.value is not None


class DecisionTreeClassifier:
    """
    Decision tree classifier that splits on information gain.

    Recursively partitions the feature space by selecting the split
    that maximizes information gain at each node. Leaf nodes predict
    the majority class of training samples that reach them.

    Parameters
    ----------
    max_depth : int
        How deep the tree can grow before stopping.
    min_samples_split : int
        Minimum number of samples required to split a node.

    Attributes
    ----------
    root : Node
        Root node of the fitted decision tree.

    Example
    -------
    >>> model = DecisionTreeClassifier(max_depth=5)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> print(model.score(X_test, y_test))
    """

    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth                  # how deep the tree can grow
        self.min_samples_split = min_samples_split  # dont split if too few samples
        self.root = None                            # tree gets built during fit

    def fit(self, X, y):
        """
        Build the decision tree from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : DecisionTreeClassifier
            Fitted model instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        # start growing the tree recursively from the top
        self.root = self._grow_tree(X, y)
        return self

    def _entropy(self, y):
        """
        Compute Shannon entropy of a label array.

        Parameters
        ----------
        y : np.ndarray
            Array of class labels.

        Returns
        -------
        float
            Entropy value. Zero means all labels are the same.
        """
        # measure how mixed up the labels are in this node
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        # add small value to avoid log(0)
        return -np.sum(probs * np.log2(probs + 1e-10))

    def _information_gain(self, y, left_y, right_y):
        """
        Compute information gain from splitting y into left_y and right_y.

        Parameters
        ----------
        y : np.ndarray
            Labels before the split.
        left_y : np.ndarray
            Labels in the left child node.
        right_y : np.ndarray
            Labels in the right child node.

        Returns
        -------
        float
            Reduction in entropy achieved by this split.
        """
        # how much does this split reduce the entropy
        n = len(y)
        parent_entropy = self._entropy(y)
        left_entropy = (len(left_y) / n) * self._entropy(left_y)
        right_entropy = (len(right_y) / n) * self._entropy(right_y)
        return parent_entropy - left_entropy - right_entropy

    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split on.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix at current node.
        y : np.ndarray of shape (n_samples,)
            Labels at current node.

        Returns
        -------
        best_feature : int or None
            Index of the best feature to split on.
        best_threshold : float or None
            Best threshold value for the split.
        """
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
                gain = self._information_gain(
                    y, y[left_mask], y[right_mask])
                # save this split if its the best one so far
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix at current node.
        y : np.ndarray of shape (n_samples,)
            Labels at current node.
        depth : int
            Current depth in the tree.

        Returns
        -------
        Node
            Root node of the subtree grown from this point.
        """
        # base cases: tree is too deep, not enough samples, or all same class
        if (depth >= self.max_depth or
                len(y) < self.min_samples_split or
                len(np.unique(y)) == 1):
            # return a leaf with the most common class
            value, counts = np.unique(y, return_counts=True)
            return Node(value=value[np.argmax(counts)])
        feature, threshold = self._best_split(X, y)
        # if no good split found just return a leaf
        if feature is None:
            value, counts = np.unique(y, return_counts=True)
            return Node(value=value[np.argmax(counts)])
        # split the data and keep growing each side
        left_mask = X[:, feature] <= threshold
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature=feature, threshold=threshold,
                    left=left, right=right)

    def _predict_single(self, x, node):
        """
        Predict the class label for a single sample by traversing the tree.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            A single input sample.
        node : Node
            Current node in the traversal.

        Returns
        -------
        int or str
            Predicted class label at the leaf node.
        """
        # walk down the tree until we hit a leaf
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X):
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_single(x, self.root) for x in X])

    def score(self, X, y):
        """
        Compute classification accuracy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
        y : array-like of shape (n_samples,)
            True class labels.

        Returns
        -------
        accuracy : float
            Fraction of correctly classified samples.
        """
        return np.mean(self.predict(X) == np.asarray(y))