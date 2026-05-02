"""Label Propagation implemented from scratch."""
import numpy as np


class LabelPropagation:
    """
    Semi-supervised label propagation algorithm.

    Spreads labels from a small set of labeled points to unlabeled points
    by building a similarity graph using the RBF kernel and iteratively
    propagating labels through the graph. Labeled points act as sources
    that continuously inject their labels, while unlabeled points absorb
    labels from their neighbors.

    Widely used for community detection in networks and semi-supervised
    classification where only a fraction of samples are labeled.

    Parameters
    ----------
    max_iter : int
        How many times to propagate labels before stopping.
    tol : float
        Stop early if labels stop changing by more than this amount.
    gamma : float
        Controls how quickly similarity drops off with distance.
        Higher gamma makes the similarity more local.

    Attributes
    ----------
    labels_ : np.ndarray of shape (n_samples,)
        Final predicted labels for every point including unlabeled ones.

    Example
    -------
    >>> model = LabelPropagation(gamma=1.0, max_iter=1000)
    >>> labels = model.fit_predict(X, y_partial)
    >>> print(model.labels_)
    """

    def __init__(self, max_iter=1000, tol=1e-3, gamma=20):
        self.max_iter = max_iter  # how many times to propagate before stopping
        self.tol = tol            # stop early if labels stop changing by this much
        self.gamma = gamma        # controls how quickly similarity drops off with distance
        self.labels_ = None       # final predicted labels for every point

    def fit(self, X, y):
        """
        Run label propagation to assign labels to all points.

        Labeled points (y != -1) act as fixed sources. Unlabeled points
        (y == -1) receive labels by absorbing from their neighbors through
        the similarity graph.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for all points, labeled and unlabeled.
        y : array-like of shape (n_samples,)
            Labels where -1 means the point is unlabeled.

        Returns
        -------
        self : LabelPropagation
            Fitted model instance with labels_ set.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n = len(X)
        # figure out what classes exist from the labeled points only
        classes = np.unique(y[y != -1])
        n_classes = len(classes)
        class_map = {c: i for i, c in enumerate(classes)}

        # build a similarity matrix using the RBF kernel
        # points that are close together get high weights
        # higher gamma makes the similarity drop off faster with distance
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = X[i] - X[j]
                    W[i, j] = np.exp(-self.gamma * np.dot(diff, diff))

        # normalize each row so weights sum to 1, like transition probabilities
        row_sums = W.sum(axis=1, keepdims=True)
        W = W / np.where(row_sums == 0, 1, row_sums)

        # set up the label matrix, labeled points get a 1 in their class column
        F = np.zeros((n, n_classes))
        labeled = y != -1
        for i in range(n):
            if labeled[i]:
                F[i, class_map[y[i]]] = 1.0
        # save the initial labels so we can keep resetting them each iteration
        F_init = F[labeled].copy()
        labeled_indices = np.where(labeled)[0]

        # keep spreading labels through the graph until convergence
        for _ in range(self.max_iter):
            F_new = W @ F
            # dont let the labeled points change, force them back each time
            F_new[labeled_indices] = F_init
            # check if labels have basically stopped changing
            if np.max(np.abs(F_new - F)) < self.tol:
                break
            F = F_new

        # each point gets the class it ended up with the most score for
        self.labels_ = np.array([classes[np.argmax(F[i])] for i in range(n)])
        return self

    def fit_predict(self, X, y):
        """
        Run label propagation and return predicted labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for all points.
        y : array-like of shape (n_samples,)
            Labels where -1 means the point is unlabeled.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Predicted labels for all points.
        """
        self.fit(X, y)
        return self.labels_