"""Logistic regression implemented from scratch using gradient descent."""
import numpy as np


class LogisticRegression:
    """Binary logistic regression, outputs probabilities using the sigmoid function."""

    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate  # how big each update step is
        self.max_iter = max_iter            # how many times to loop through gradient descent
        self.coef_ = None                   # feature weights learned during training
        self.intercept_ = None              # bias term learned during training

    def _sigmoid(self, z):
        # squashes any value into a probability between 0 and 1
        # clip z to prevent overflow when exp gets very large or very small
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        # start everything at zero before training
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for _ in range(self.max_iter):
            # compute the linear combination and pass through sigmoid
            z = X @ self.coef_ + self.intercept_
            y_pred = self._sigmoid(z)
            # difference between predicted probability and true label
            error = y_pred - y
            # update weights and bias using gradient of cross entropy loss
            self.coef_ -= self.learning_rate * (X.T @ error) / n_samples
            self.intercept_ -= self.learning_rate * np.mean(error)

        return self

    def predict_proba(self, X):
        # run the forward pass and return raw probabilities
        X = np.asarray(X, dtype=float)
        return self._sigmoid(X @ self.coef_ + self.intercept_)

    def predict(self, X):
        # anything above 0.5 probability gets predicted as class 1
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))