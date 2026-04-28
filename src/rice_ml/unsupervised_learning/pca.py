"""Principal Component Analysis implemented from scratch."""
import numpy as np


class PCA:
    """PCA for dimensionality reduction using eigendecomposition of the covariance matrix."""

    def __init__(self, n_components):
        self.n_components = n_components          # how many components to keep
        self.components_ = None                   # the principal component directions
        self.mean_ = None                         # mean of the training data, needed for centering
        self.explained_variance_ = None           # how much variance each component captures
        self.explained_variance_ratio_ = None     # same but as a fraction of total variance

    def fit(self, X):
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
        # project the data onto our principal components
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        # fit and transform in one step, useful shortcut for training data
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed):
        # go back from the compressed space to approximate the original data
        # wont be exactly the same if we dropped components
        return X_transformed @ self.components_ + self.mean_