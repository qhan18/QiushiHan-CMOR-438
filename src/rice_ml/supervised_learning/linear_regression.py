"""Linear regression implemented from scratch using OLS, Ridge, and Gradient Descent."""
import numpy as np


class LinearRegression:
    """
    Linear Regression with three solvers to choose from.

    Implements ordinary least squares, ridge regression with L2
    regularization, and iterative gradient descent. All three methods
    learn a linear mapping from input features to a continuous target.

    Parameters
    ----------
    method : str
        Which solver to use. One of 'ols', 'ridge', or 'gd'.
    alpha : float
        Regularization strength for ridge regression. Larger values
        penalize large weights more heavily.
    learning_rate : float
        Step size for gradient descent updates.
    max_iter : int
        Number of gradient descent iterations to run.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Feature weights learned during fit.
    intercept_ : float
        Bias term learned during fit.

    Example
    -------
    >>> model = LinearRegression(method='ols')
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> print(model.score(X_test, y_test))
    """

    def __init__(self, method='ols', alpha=1.0,
                 learning_rate=0.01, max_iter=1000):
        self.method = method               # which solver to use
        self.alpha = alpha                 # regularization strength, only used for ridge
        self.learning_rate = learning_rate # how big each gradient descent step is
        self.max_iter = max_iter           # how many GD iterations to run
        self.coef_ = None                  # feature weights learned during fit
        self.intercept_ = None             # bias term learned during fit

    def fit(self, X, y):
        """
        Fit the linear regression model to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : LinearRegression
            Fitted model instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # call the right solver based on what method was chosen
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
        """Fit using the closed-form normal equation."""
        # prepend a column of ones to handle the bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # normal equation gives the exact closed form solution
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

    def _fit_ridge(self, X, y):
        """Fit using ridge regression with L2 regularization."""
        # prepend a column of ones for the bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        n_features = X_b.shape[1]
        # set up identity matrix but dont regularize the bias term
        I = np.eye(n_features)
        I[0, 0] = 0
        # ridge version of the normal equation adds alpha * I to stabilize
        theta = np.linalg.pinv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

    def _fit_gd(self, X, y):
        """Fit using iterative gradient descent on MSE loss."""
        n_samples, n_features = X.shape
        # start weights and bias at zero
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        for _ in range(self.max_iter):
            # make predictions with current weights
            y_pred = X @ self.coef_ + self.intercept_
            error = y_pred - y
            # nudge weights in the direction that reduces MSE
            self.coef_ -= self.learning_rate * (X.T @ error) / n_samples
            self.intercept_ -= self.learning_rate * np.mean(error)

    def predict(self, X):
        """
        Predict target values for input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted target values.
        """
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """
        Compute R squared coefficient of determination.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
        y : array-like of shape (n_samples,)
            True target values.

        Returns
        -------
        r2 : float
            R squared score. 1.0 means perfect prediction.
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot