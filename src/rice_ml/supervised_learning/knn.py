"""K-Nearest Neighbors implemented from scratch."""
import numpy as np


class KNN:
    """K-Nearest Neighbors classifier using Euclidean distance."""

    def __init__(self, k=3):
        self.k = k          # number of nearest neighbors to consider
        self.X_train = None # stored training features
        self.y_train = None # stored training labels

    def fit(self, X, y):
        # KNN has no training phase — just store the training data
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def _euclidean_distance(self, x1, x2):
        # compute straight-line distance between two points
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # predict each test point individually
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        # compute distance from x to every training point
        distances = [self._euclidean_distance(x, x_train)
                     for x_train in self.X_train]
        # get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        # return majority class among neighbors
        values, counts = np.unique(k_labels, return_counts=True)
        return values[np.argmax(counts)]

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))