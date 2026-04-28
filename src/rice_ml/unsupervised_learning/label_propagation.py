"""Label Propagation implemented from scratch."""
import numpy as np


class LabelPropagation:
    """Semi-supervised algorithm that spreads labels from labeled to unlabeled points."""

    def __init__(self, max_iter=1000, tol=1e-3, gamma=20):
        self.max_iter = max_iter  # how many times to propagate before stopping
        self.tol = tol            # stop early if labels stop changing by this much
        self.gamma = gamma        # controls how quickly similarity drops off with distance
        self.labels_ = None       # final predicted labels for every point

    def fit(self, X, y):
        """
        X: feature matrix
        y: labels where -1 means the point is unlabeled
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
        self.fit(X, y)
        return self.labels_