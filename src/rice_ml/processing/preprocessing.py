"""Preprocessing utilities implemented from scratch."""
import numpy as np


class StandardScaler:
    """Standardize features by removing mean and scaling to unit variance."""

    def __init__(self):
        self.mean_ = None  # mean of each feature
        self.std_ = None   # standard deviation of each feature

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # replace zero std with 1 to avoid division by zero
        self.std_ = np.where(self.std_ == 0, 1, self.std_)
        return self

    def transform(self, X):
        # subtract mean and divide by std for each feature
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        # reverse the scaling to recover original values
        return X_scaled * self.std_ + self.mean_


class MinMaxScaler:
    """Scale features to a given range (default 0 to 1)."""

    def __init__(self):
        self.min_ = None  # minimum of each feature
        self.max_ = None  # maximum of each feature

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        # avoid division by zero for constant features
        denom = np.where(self.max_ - self.min_ == 0, 1, self.max_ - self.min_)
        return (X - self.min_) / denom

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split arrays into random train and test subsets."""
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    n = len(X)
    n_test = int(n * test_size)
    # shuffle indices randomly
    indices = rng.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]