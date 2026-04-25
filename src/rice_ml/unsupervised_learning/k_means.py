"""K-Means Clustering implemented from scratch."""
import numpy as np


class KMeans:
    """K-Means clustering using Lloyd's algorithm."""

    def __init__(self, k=3, max_iter=300, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X), self.k, replace=False)
        self.centroids = X[indices].copy()

        for _ in range(self.max_iter):
            labels = self._assign_clusters(X)
            new_centroids = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i)
                else self.centroids[i]
                for i in range(self.k)
            ])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        self.labels_ = self._assign_clusters(X)
        return self

    def _assign_clusters(self, X):
        distances = np.array([
            np.linalg.norm(X - centroid, axis=1)
            for centroid in self.centroids
        ])
        return np.argmin(distances, axis=0)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self._assign_clusters(X)

    def inertia(self, X):
        X = np.asarray(X, dtype=float)
        labels = self.predict(X)
        return sum(
            np.sum((X[labels == i] - self.centroids[i]) ** 2)
            for i in range(self.k)
        )