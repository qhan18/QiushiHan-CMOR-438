"""Label Propagation implemented from scratch."""
import numpy as np


class LabelPropagation:
    """Semi-supervised label propagation algorithm."""

    def __init__(self, max_iter=1000, tol=1e-3, gamma=20):
        self.max_iter = max_iter
        self.tol = tol
        self.gamma = gamma
        self.labels_ = None

    def fit(self, X, y):
        """
        X: feature matrix
        y: labels where -1 indicates unlabeled points
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n = len(X)
        classes = np.unique(y[y != -1])
        n_classes = len(classes)
        class_map = {c: i for i, c in enumerate(classes)}

        # Build weight matrix using RBF kernel
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = X[i] - X[j]
                    W[i, j] = np.exp(-self.gamma * np.dot(diff, diff))

        # Row normalize
        row_sums = W.sum(axis=1, keepdims=True)
        W = W / np.where(row_sums == 0, 1, row_sums)

        # Initialize label matrix
        F = np.zeros((n, n_classes))
        labeled = y != -1
        for i in range(n):
            if labeled[i]:
                F[i, class_map[y[i]]] = 1.0
        F_init = F[labeled].copy()
        labeled_indices = np.where(labeled)[0]

        # Propagate
        for _ in range(self.max_iter):
            F_new = W @ F
            F_new[labeled_indices] = F_init
            if np.max(np.abs(F_new - F)) < self.tol:
                break
            F = F_new

        self.labels_ = np.array([classes[np.argmax(F[i])] for i in range(n)])
        return self

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.labels_