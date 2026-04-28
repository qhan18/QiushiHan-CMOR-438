"""Multilayer Perceptron implemented from scratch using backpropagation."""
import numpy as np


class MLP:
    """Feedforward neural network with one hidden layer, trained with backprop."""

    def __init__(self, hidden_size=64, learning_rate=0.01, max_iter=1000):
        self.hidden_size = hidden_size      # how many neurons in the hidden layer
        self.learning_rate = learning_rate  # how big each gradient step is
        self.max_iter = max_iter            # how many training iterations to run
        self.W1 = None   # weight matrix from input to hidden layer
        self.b1 = None   # bias vector for hidden layer
        self.W2 = None   # weight matrix from hidden to output layer
        self.b2 = None   # bias vector for output layer

    def _sigmoid(self, z):
        # clip z to avoid overflow when computing exp of large numbers
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _sigmoid_deriv(self, z):
        # need this for backprop, derivative of sigmoid is s * (1 - s)
        s = self._sigmoid(z)
        return s * (1 - s)

    def fit(self, X, y):
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
        # just run the forward pass, no weight updates during prediction
        X = np.asarray(X, dtype=float)
        a1 = self._sigmoid(X @ self.W1 + self.b1)
        return self._sigmoid(a1 @ self.W2 + self.b2).flatten()

    def predict(self, X):
        # anything above 0.5 gets predicted as class 1
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))