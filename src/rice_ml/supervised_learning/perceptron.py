"""Perceptron implemented from scratch."""
import numpy as np


class Perceptron:
    """Single layer Perceptron for binary classification, the simplest neural network."""

    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate  # how much to adjust weights on each mistake
        self.max_iter = max_iter            # how many full passes through the training data
        self.coef_ = None                   # feature weights, learned during training
        self.intercept_ = None              # bias term, learned during training

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        # start with all weights and bias at zero
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for _ in range(self.max_iter):
            for i, x_i in enumerate(X):
                # get current prediction for this sample
                y_pred = self._predict_single(x_i)
                # only update if we got it wrong, no update needed if correct
                update = self.learning_rate * (y[i] - y_pred)
                self.coef_ += update * x_i
                self.intercept_ += update

        return self

    def _predict_single(self, x):
        # if the dot product plus bias is >= 0 predict 1, otherwise predict 0
        return 1 if np.dot(x, self.coef_) + self.intercept_ >= 0 else 0

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # run prediction for each sample individually
        return np.array([self._predict_single(x) for x in X])

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))