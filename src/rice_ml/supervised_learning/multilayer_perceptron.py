"""Multilayer Perceptron implemented from scratch using backpropagation."""
import numpy as np


class MLP:
    """
    Feedforward neural network with one hidden layer.

    Trained using backpropagation and batch gradient descent to minimize
    binary cross-entropy loss. Uses sigmoid activations in both the hidden
    and output layers. Supports binary classification tasks.

    Parameters
    ----------
    hidden_size : int
        How many neurons in the hidden layer.
    learning_rate : float
        How big each gradient descent step is.
    max_iter : int
        How many training iterations to run.

    Attributes
    ----------
    W1 : np.ndarray of shape (n_features, hidden_size)
        Weight matrix from input to hidden layer.
    b1 : np.ndarray of shape (hidden_size,)
        Bias vector for hidden layer.
    W2 : np.ndarray of shape (hidden_size, 1)
        Weight matrix from hidden to output layer.
    b2 : np.ndarray of shape (1,)
        Bias vector for output layer.

    Example
    -------
    >>> model = MLP(hidden_size=16, learning_rate=0.1, max_iter=1000)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> print(model.score(X_test, y_test))
    """

    def __init__(self, hidden_size=64, learning_rate=0.01, max_iter=1000):
        self.hidden_size = hidden_size      # how many neurons in the hidden layer
        self.learning_rate = learning_rate  # how big each gradient step is
        self.max_iter = max_iter            # how many training iterations to run
        self.W1 = None   # weight matrix from input to hidden layer
        self.b1 = None   # bias vector for hidden layer
        self.W2 = None   # weight matrix from hidden to output layer
        self.b2 = None   # bias vector for output layer

    def _sigmoid(self, z):
        """
        Apply sigmoid activation function element-wise.

        Parameters
        ----------
        z : np.ndarray
            Input values to activate.

        Returns
        -------
        np.ndarray
            Values squashed to range (0, 1).
        """
        # clip z to avoid overflow when computing exp of large numbers
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _sigmoid_deriv(self, z):
        """
        Compute the derivative of the sigmoid function.

        Used during backpropagation to compute gradients.

        Parameters
        ----------
        z : np.ndarray
            Input values before activation.

        Returns
        -------
        np.ndarray
            Derivative of sigmoid at each input value.
        """
        # need this for backprop, derivative of sigmoid is s * (1 - s)
        s = self._sigmoid(z)
        return s * (1 - s)

    def fit(self, X, y):
        """
        Train the MLP using backpropagation and gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Binary target labels (0 or 1).

        Returns
        -------
        self : MLP
            Fitted model instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        rng = np.random.default_rng(42)

        # small random init is important, all zeros would cause symmetry problems
        self.W1 = rng.standard_normal((n_features, self.hidden_size)) * 0.01
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = rng.standard_normal((self.hidden_size, 1)) * 0.01
        self.b2 = np.zeros(1)

        for _ in range(self.max_iter):
            # forward pass through the hidden layer
            z1 = X @ self.W1 + self.b1
            a1 = self._sigmoid(z1)
            # forward pass through the output layer
            z2 = a1 @ self.W2 + self.b2
            a2 = self._sigmoid(z2).flatten()

            # backprop: compute how much each weight contributed to the error
            error = a2 - y
            dW2 = (a1.T @ (error * self._sigmoid_deriv(z2.flatten())
                           ).reshape(-1, 1)) / n_samples
            db2 = np.mean(error * self._sigmoid_deriv(z2.flatten()))
            dz1 = (error * self._sigmoid_deriv(z2.flatten())
                   ).reshape(-1, 1) * self.W2.T * self._sigmoid_deriv(z1)
            dW1 = X.T @ dz1 / n_samples
            db1 = np.mean(dz1, axis=0)

            # update weights and biases in the direction that reduces loss
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

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
        # just run the forward pass, no weight updates during prediction
        X = np.asarray(X, dtype=float)
        a1 = self._sigmoid(X @ self.W1 + self.b1)
        return self._sigmoid(a1 @ self.W2 + self.b2).flatten()

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
        # anything above 0.5 gets predicted as class 1
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