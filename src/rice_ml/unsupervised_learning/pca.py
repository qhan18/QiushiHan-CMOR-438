"""Principal Component Analysis implemented from scratch."""
import numpy as np


class PCA:
    """
    Principal Component Analysis using eigendecomposition.

    Reduces dimensionality by projecting data onto the directions of
    maximum variance in the dataset. These directions, called principal
    components, are the eigenvectors of the covariance matrix sorted
    by descending eigenvalue.

    Parameters
    ----------
    n_components : int
        How many principal components to keep after dimensionality reduction.

    Attributes
    ----------
    components_ : np.ndarray of shape (n_components, n_features)
        Principal component directions, sorted by explained variance.
    mean_ : np.ndarray of shape (n_features,)
        Mean of the training data, used for centering.
    explained_variance_ : np.ndarray of shape (n_components,)
        Amount of variance captured by each principal component.
    explained_variance_ratio_ : np.ndarray of shape (n_components,)
        Fraction of total variance captured by each principal component.

    Example
    -------
    >>> pca = PCA(n_components=2)
    >>> X_reduced = pca.fit_transform(X)
    >>> print(pca.explained_variance_ratio_)
    >>> X_reconstructed = pca.inverse_transform(X_reduced)
    """

    def __init__(self, n_components):
        self.n_components = n_components          # how many components to keep
        self.components_ = None                   # the principal component directions
        self.mean_ = None                         # mean of the training data, needed for centering
        self.explained_variance_ = None           # how much variance each component captures
        self.explained_variance_ratio_ = None     # same but as a fraction of total variance

    def fit(self, X):
        """
        Fit PCA by computing principal components from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data to compute components from.

        Returns
        -------
        self : PCA
            Fitted model instance.
        """
        X = np.asarray(X, dtype=float)
        # center the data first, PCA requires zero mean
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        # compute covariance matrix to understand how features vary together
        cov_matrix = np.cov(X_centered, rowvar=False)
        # find the eigenvectors and eigenvalues of the covariance matrix
        # eigenvectors are the principal component directions
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # sort from largest to smallest eigenvalue so we keep the most important components first
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        # grab only the top n_components eigenvectors
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        # what fraction of total variance do our components explain
        self.explained_variance_ratio_ = (
            self.explained_variance_ / np.sum(eigenvalues)
        )
        return self

    def transform(self, X):
        """
        Project data onto the principal components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to project into the reduced space.

        Returns
        -------
        X_reduced : np.ndarray of shape (n_samples, n_components)
            Data projected onto principal components.
        """
        # project the data onto our principal components
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        """
        Fit PCA and project data in one step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data to fit and transform.

        Returns
        -------
        X_reduced : np.ndarray of shape (n_samples, n_components)
            Data projected onto principal components.
        """
        # fit and transform in one step, useful shortcut for training data
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed):
        """
        Reconstruct approximate original data from compressed representation.

        The reconstruction will not be exact if components were dropped,
        since information was lost during dimensionality reduction.

        Parameters
        ----------
        X_transformed : array-like of shape (n_samples, n_components)
            Data in the reduced principal component space.

        Returns
        -------
        X_reconstructed : np.ndarray of shape (n_samples, n_features)
            Approximate reconstruction in the original feature space.
        """
        # go back from the compressed space to approximate the original data
        # wont be exactly the same if we dropped components
        return X_transformed @ self.components_ + self.mean_