"""Principal Component Analysis implemented from scratch."""
import numpy as np


class PCA:
    """Principal Component Analysis using eigendecomposition."""

    def __init__(self, n_components):
        self.n_components = n_components          # number of components to keep
        self.components_ = None                   # principal component vectors
        self.mean_ = None                         # mean of training data
        self.explained_variance_ = None           # variance of each component
        self.explained_variance_ratio_ = None     # fraction of total variance

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        # center the data by subtracting the mean
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        # compute the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        # compute eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # sort by descending eigenvalue (most variance first)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        # keep only the top n_components eigenvectors
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = (
            self.explained_variance_ / np.sum(eigenvalues)
        )
        return self

    def transform(self, X):
        # project data onto principal components
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed):
        # reconstruct approximate original data from compressed representation
        return X_transformed @ self.components_ + self.mean_