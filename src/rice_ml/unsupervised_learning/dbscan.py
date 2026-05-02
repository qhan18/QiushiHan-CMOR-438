"""DBSCAN clustering implemented from scratch."""
import numpy as np


class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise.

    Groups points into clusters based on density rather than distance
    to a centroid. Can discover clusters of arbitrary shape and
    automatically identifies noise points that don't belong to any
    cluster. Points are labeled -1 if they are considered noise.

    Parameters
    ----------
    eps : float
        How far apart two points can be to be considered neighbors.
        Smaller values create more clusters and more noise points.
    min_samples : int
        Minimum number of neighbors a point needs to be a core point.
        Higher values require denser regions to form clusters.

    Attributes
    ----------
    labels_ : np.ndarray of shape (n_samples,)
        Cluster assignment for each point. Noise points are labeled -1.

    Example
    -------
    >>> model = DBSCAN(eps=0.3, min_samples=5)
    >>> model.fit(X)
    >>> print(model.labels_)
    >>> labels = model.fit_predict(X)
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps                  # how far apart two points can be to be considered neighbors
        self.min_samples = min_samples  # minimum neighbors needed to be a core point
        self.labels_ = None             # cluster assignments, noise points get labeled -1

    def fit(self, X):
        """
        Run DBSCAN clustering on input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to cluster.

        Returns
        -------
        self : DBSCAN
            Fitted model instance with labels_ set.
        """
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
        """
        Find all points within eps distance of point i.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Full dataset.
        i : int
            Index of the query point.

        Returns
        -------
        np.ndarray
            Indices of all neighboring points within eps distance.
        """
        # compute distances from point i to all other points
        distances = np.linalg.norm(X - X[i], axis=1)
        # return indices of all points within eps radius
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, i, neighbors, cluster_id, visited):
        """
        Expand a cluster starting from core point i.

        Iteratively adds reachable points to the cluster by checking
        if each neighbor is also a core point and adding its neighbors
        to the expansion queue.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Full dataset.
        i : int
            Index of the starting core point.
        neighbors : np.ndarray
            Initial neighbors of point i.
        cluster_id : int
            ID of the cluster being expanded.
        visited : np.ndarray of bool
            Tracks which points have already been processed.
        """
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
        """
        Run clustering and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to cluster.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster assignments. Noise points are labeled -1.
        """
        self.fit(X)
        return self.labels_