"""K-Nearest Neighbors implemented from scratch."""
import numpy as np


class KNN:
    """KNN classifier, predicts based on the k closest training points."""

    def __init__(self, k=3):
        self.k = k          # how many neighbors to look at
        self.X_train = None # store training data during fit
        self.y_train = None # store training labels during fit

    def fit(self, X, y):
        # KNN doesnt actually learn anything, just memorizes the training data
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def _euclidean_distance(self, x1, x2):
        # standard straight line distance between two points
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # run prediction for each test point one at a time
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        # figure out how far this point is from every training point
        distances = [self._euclidean_distance(x, x_train)
                     for x_train in self.X_train]
        # sort and grab the k closest ones
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        # whichever class shows up most among the neighbors wins
        values, counts = np.unique(k_labels, return_counts=True)
        return values[np.argmax(counts)]

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))