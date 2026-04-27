"""Logistic regression implemented from scratch using gradient descent."""
import numpy as np


class LogisticRegression:
    """Binary logistic regression using gradient descent."""

    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate  # step size for gradient descent
        self.max_iter = max_iter            # maximum number of iterations
        self.coef_ = None                   # learned feature weights
        self.intercept_ = None              # learned bias term

    def _sigmoid(self, z):
        # clip to avoid numerical overflow in exp
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        # initialize weights and bias to zero
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for _ in range(self.max_iter):
            # forward pass: compute predicted probabilities
            z = X @ self.coef_ + self.intercept_
            y_pred = self._sigmoid(z)
            # compute gradient of binary cross-entropy loss
            error = y_pred - y
            # update weights and bias
            self.coef_ -= self.learning_rate * (X.T @ error) / n_samples
            self.intercept_ -= self.learning_rate * np.mean(error)

        return self

    def predict_proba(self, X):
        # return predicted probability of positive class
        X = np.asarray(X, dtype=float)
        return self._sigmoid(X @ self.coef_ + self.intercept_)

    def predict(self, X):
        # threshold probabilities at 0.5 to get class labels
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))