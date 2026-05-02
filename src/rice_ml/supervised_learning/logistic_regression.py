"""Logistic regression implemented from scratch using gradient descent."""
import numpy as np


class LogisticRegression:
    """
    Binary logistic regression using gradient descent.

    Models the probability of class membership using the sigmoid function
    and trains by minimizing binary cross-entropy loss via batch gradient
    descent. Outputs probabilities between 0 and 1, thresholded at 0.5
    for binary predictions.

    Parameters
    ----------
    learning_rate : float
        How big each gradient descent update step is.
    max_iter : int
        How many times to loop through gradient descent.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Feature weights learned during training.
    intercept_ : float
        Bias term learned during training.

    Example
    -------
    >>> model = LogisticRegression(learning_rate=0.1, max_iter=1000)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> print(model.score(X_test, y_test))
    """

    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate  # how big each update step is
        self.max_iter = max_iter            # how many times to loop through gradient descent
        self.coef_ = None                   # feature weights learned during training
        self.intercept_ = None              # bias term learned during training

    def _sigmoid(self, z):
        """
        Apply sigmoid activation function.

        Squashes any value into a probability between 0 and 1.
        Clips input to prevent numerical overflow.

        Parameters
        ----------
        z : np.ndarray
            Linear combination of inputs and weights.

        Returns
        -------
        np.ndarray
            Probabilities between 0 and 1.
        """
        # clip z to prevent overflow when exp gets very large or very small
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        """
        Fit logistic regression using gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Binary target labels (0 or 1).

        Returns
        -------
        self : LogisticRegression
            Fitted model instance.
        """
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
        """
        Predict class membership probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        proba : np.ndarray of shape (n_samples,)
            Predicted probability of belonging to class 1.
        """
        # run the forward pass and return raw probabilities
        X = np.asarray(X, dtype=float)
        return self._sigmoid(X @ self.coef_ + self.intercept_)

    def predict(self, X):
        """
        Predict binary class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        # anything above 0.5 probability gets predicted as class 1
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X, y):
        """
        Compute classification accuracy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
        y : array-like of shape (n_samples,)
            True binary labels.

        Returns
        -------
        accuracy : float
            Fraction of correctly classified samples.
        """
        return np.mean(self.predict(X) == np.asarray(y))