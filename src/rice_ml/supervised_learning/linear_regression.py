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
        self.method = method              # which solver to use
        self.alpha = alpha                # regularization strength for ridge
        self.learning_rate = learning_rate  # step size for gradient descent
        self.max_iter = max_iter          # maximum number of GD iterations
        self.coef_ = None                 # learned feature weights
        self.intercept_ = None            # learned bias term

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # dispatch to the appropriate solver
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
        # add bias column of ones to X
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # solve normal equation: theta = (X^T X)^-1 X^T y
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

    def _fit_ridge(self, X, y):
        # add bias column
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        n_features = X_b.shape[1]
        # identity matrix with bias term excluded from regularization
        I = np.eye(n_features)
        I[0, 0] = 0
        # solve ridge normal equation: theta = (X^T X + alpha*I)^-1 X^T y
        theta = np.linalg.pinv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

    def _fit_gd(self, X, y):
        n_samples, n_features = X.shape
        # initialize weights and bias to zero
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        for _ in range(self.max_iter):
            # compute predictions and error
            y_pred = X @ self.coef_ + self.intercept_
            error = y_pred - y
            # update weights using gradient of MSE loss
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