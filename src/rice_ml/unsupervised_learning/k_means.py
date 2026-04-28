"""K-Means Clustering implemented from scratch."""
import numpy as np


class KMeans:
    """K-Means clustering using Lloyd's algorithm, groups data into k clusters."""

    def __init__(self, k=3, max_iter=300, random_state=42):
        self.k = k                        # how many clusters to find
        self.max_iter = max_iter          # max number of iterations before we stop
        self.random_state = random_state  # for reproducibility
        self.centroids = None             # the cluster centers, learned during fit
        self.labels_ = None               # which cluster each point belongs to

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        # pick k random data points as our starting centroids
        indices = rng.choice(len(X), self.k, replace=False)
        self.centroids = X[indices].copy()

        for _ in range(self.max_iter):
            # figure out which centroid each point is closest to
            labels = self._assign_clusters(X)
            # move each centroid to the mean of its assigned points
            new_centroids = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i)
                else self.centroids[i]
                for i in range(self.k)
            ])
            # if centroids didnt move much we can stop early
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        # do one final assignment with the converged centroids
        self.labels_ = self._assign_clusters(X)
        return self

    def _assign_clusters(self, X):
        # compute how far each point is from each centroid
        distances = np.array([
            np.linalg.norm(X - centroid, axis=1)
            for centroid in self.centroids
        ])
        # each point gets assigned to its nearest centroid
        return np.argmin(distances, axis=0)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self._assign_clusters(X)

    def inertia(self, X):
        # sum up squared distances from each point to its assigned centroid
        # lower inertia means tighter clusters
        X = np.asarray(X, dtype=float)
        labels = self.predict(X)
        return sum(
            np.sum((X[labels == i] - self.centroids[i]) ** 2)
            for i in range(self.k)
        )