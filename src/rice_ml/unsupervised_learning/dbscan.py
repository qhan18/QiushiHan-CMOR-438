"""DBSCAN clustering implemented from scratch."""
import numpy as np


class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise."""

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        self.labels_ = np.full(n, -1)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._get_neighbors(X, i)
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
                continue
            self._expand_cluster(X, i, neighbors, cluster_id, visited)
            cluster_id += 1

        return self

    def _get_neighbors(self, X, i):
        distances = np.linalg.norm(X - X[i], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, i, neighbors, cluster_id, visited):
        self.labels_[i] = cluster_id
        neighbors = list(neighbors)
        idx = 0
        while idx < len(neighbors):
            j = neighbors[idx]
            if not visited[j]:
                visited[j] = True
                new_neighbors = self._get_neighbors(X, j)
                if len(new_neighbors) >= self.min_samples:
                    neighbors += [n for n in new_neighbors
                                  if n not in neighbors]
            if self.labels_[j] == -1:
                self.labels_[j] = cluster_id
            idx += 1

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_