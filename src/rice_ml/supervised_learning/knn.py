"""K-Nearest Neighbors implemented from scratch."""
import numpy as np


class KNN:
    """K-Nearest Neighbors classifier."""

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        distances = [self._euclidean_distance(x, x_train)
                     for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        values, counts = np.unique(k_labels, return_counts=True)
        return values[np.argmax(counts)]

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))