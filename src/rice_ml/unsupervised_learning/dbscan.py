"""DBSCAN clustering implemented from scratch."""
import numpy as np


class DBSCAN:
    """Density based clustering that can find arbitrary shapes and label noise points."""

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps                  # how far apart two points can be to be considered neighbors
        self.min_samples = min_samples  # minimum neighbors needed to be a core point
        self.labels_ = None             # cluster assignments, noise points get labeled -1

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        # start by assuming everything is noise, we will assign clusters as we go
        self.labels_ = np.full(n, -1)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        for i in range(n):
            # dont process a point we already visited
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._get_neighbors(X, i)
            # not enough nearby points so this one stays as noise
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
                continue
            # found a core point so start growing a new cluster from it
            self._expand_cluster(X, i, neighbors, cluster_id, visited)
            cluster_id += 1

        return self

    def _get_neighbors(self, X, i):
        # compute distances from point i to all other points
        distances = np.linalg.norm(X - X[i], axis=1)
        # return indices of all points within eps radius
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, i, neighbors, cluster_id, visited):
        # put the starting point in this cluster
        self.labels_[i] = cluster_id
        neighbors = list(neighbors)
        idx = 0
        while idx < len(neighbors):
            j = neighbors[idx]
            if not visited[j]:
                visited[j] = True
                # check if this neighbor is also a core point
                new_neighbors = self._get_neighbors(X, j)
                # if it is, add all its neighbors to our expansion queue
                if len(new_neighbors) >= self.min_samples:
                    neighbors += [n for n in new_neighbors
                                  if n not in neighbors]
            # add this point to the cluster if it hasnt been assigned yet
            if self.labels_[j] == -1:
                self.labels_[j] = cluster_id
            idx += 1

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_