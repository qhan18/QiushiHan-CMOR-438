"""Linear regression implemented from scratch: OLS, Ridge, and Gradient Descent."""
import numpy as np


class LinearRegression:
    """
    Linear Regression with three methods:
    - 'ols': Ordinary Least Squares (normal equation)
    - 'ridge': Ridge Regression (L2 regularization, closed form)
    - 'gd': Gradient Descent (iterative)
    """

    def __init__(self, method='ols', alpha=1.0,
                 learning_rate=0.01, max_iter=1000):
        self.method = method
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.method == 'ols':
            self._fit_ols(X, y)
        elif self.method == 'ridge':
            self._fit_ridge(X, y)
        elif self.method == 'gd':
            self._fit_gd(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return self

    def _fit_ols(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

    def _fit_ridge(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        n_features = X_b.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0  # don't regularize bias
        theta = np.linalg.pinv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

    def _fit_gd(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        for _ in range(self.max_iter):
            y_pred = X @ self.coef_ + self.intercept_
            error = y_pred - y
            self.coef_ -= self.learning_rate * (X.T @ error) / n_samples
            self.intercept_ -= self.learning_rate * np.mean(error)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """R² coefficient of determination."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot