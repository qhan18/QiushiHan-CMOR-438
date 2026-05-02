"""K-Means Clustering implemented from scratch."""
import numpy as np


class KMeans:
    """
    K-Means clustering using Lloyd's algorithm.

    Groups n samples into k clusters by iteratively assigning each point
    to its nearest centroid and recomputing centroids as the mean of
    assigned points. Stops when centroids converge or max_iter is reached.

    Parameters
    ----------
    k : int
        How many clusters to find.
    max_iter : int
        Maximum number of iterations before stopping.
    random_state : int
        Seed for random centroid initialization, ensures reproducibility.

    Attributes
    ----------
    centroids : np.ndarray of shape (k, n_features)
        Cluster centers learned during fit.
    labels_ : np.ndarray of shape (n_samples,)
        Cluster assignment for each training point.

    Example
    -------
    >>> model = KMeans(k=3, random_state=42)
    >>> model.fit(X)
    >>> print(model.labels_)
    >>> print(model.inertia(X))
    """

    def __init__(self, k=3, max_iter=300, random_state=42):
        self.k = k                        # how many clusters to find
        self.max_iter = max_iter          # max number of iterations before we stop
        self.random_state = random_state  # for reproducibility
        self.centroids = None             # the cluster centers, learned during fit
        self.labels_ = None               # which cluster each point belongs to

    def fit(self, X):
        """
        Fit K-Means to the data by running Lloyd's algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to cluster.

        Returns
        -------
        self : KMeans
            Fitted model instance.
        """
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
        """
        Assign each sample to its nearest centroid.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster index for each sample.
        """
        # compute how far each point is from each centroid
        distances = np.array([
            np.linalg.norm(X - centroid, axis=1)
            for centroid in self.centroids
        ])
        # each point gets assigned to its nearest centroid
        return np.argmin(distances, axis=0)

    def predict(self, X):
        """
        Predict cluster assignments for new samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to cluster.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster index for each sample.
        """
        X = np.asarray(X, dtype=float)
        return self._assign_clusters(X)

    def inertia(self, X):
        """
        Compute within-cluster sum of squared distances.

        Lower inertia means tighter, more compact clusters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        float
            Sum of squared distances from each point to its assigned centroid.
        """
        # sum up squared distances from each point to its assigned centroid
        X = np.asarray(X, dtype=float)
        labels = self.predict(X)
        return sum(
            np.sum((X[labels == i] - self.centroids[i]) ** 2)
            for i in range(self.k)
        )