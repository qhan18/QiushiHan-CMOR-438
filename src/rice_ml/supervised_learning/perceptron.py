"""Perceptron implemented from scratch."""
import numpy as np


class Perceptron:
    """
    Single layer Perceptron for binary classification.

    The simplest neural network, introduced by Rosenblatt in 1958.
    Learns a linear decision boundary by updating weights only when
    a sample is misclassified. Guaranteed to converge only if the
    data is linearly separable.

    Parameters
    ----------
    learning_rate : float
        How much to adjust weights on each mistake.
    max_iter : int
        How many full passes through the training data.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Feature weights learned during training.
    intercept_ : float
        Bias term learned during training.

    Example
    -------
    >>> model = Perceptron(learning_rate=0.1, max_iter=100)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> print(model.score(X_test, y_test))
    """

    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate  # how much to adjust weights on each mistake
        self.max_iter = max_iter            # how many full passes through the training data
        self.coef_ = None                   # feature weights, learned during training
        self.intercept_ = None              # bias term, learned during training

    def fit(self, X, y):
        """
        Train the perceptron using the Rosenblatt learning rule.

        Iterates through the training data multiple times, updating
        weights whenever a sample is misclassified.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Binary target labels (0 or 1).

        Returns
        -------
        self : Perceptron
            Fitted model instance.
        """
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
        """
        Predict the class label for a single sample.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            A single input sample.

        Returns
        -------
        int
            Predicted class label, either 0 or 1.
        """
        # if the dot product plus bias is >= 0 predict 1, otherwise predict 0
        return 1 if np.dot(x, self.coef_) + self.intercept_ >= 0 else 0

    def predict(self, X):
        """
        Predict binary class labels for input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        X = np.asarray(X, dtype=float)
        # run prediction for each sample individually
        return np.array([self._predict_single(x) for x in X])

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