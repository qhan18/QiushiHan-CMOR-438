"""Preprocessing utilities implemented from scratch."""
import numpy as np


class StandardScaler:
    """Standardizes features to have zero mean and unit variance."""

    def __init__(self):
        self.mean_ = None  # will store the mean of each feature after fitting
        self.std_ = None   # will store the std of each feature after fitting

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        # compute mean and std across all training samples for each feature
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # if a feature has zero variance, set std to 1 so we dont divide by zero
        self.std_ = np.where(self.std_ == 0, 1, self.std_)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        # apply z-score normalization using the stored mean and std
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        # fit and transform in one step, useful for training data
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        # undo the scaling to get back to the original feature space
        return X_scaled * self.std_ + self.mean_


class MinMaxScaler:
    """Scales features to the range [0, 1]."""

    def __init__(self):
        self.min_ = None  # minimum value of each feature
        self.max_ = None  # maximum value of each feature

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        # store min and max so we can scale new data consistently
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        # handle edge case where a feature is constant (max == min)
        denom = np.where(self.max_ - self.min_ == 0, 1, self.max_ - self.min_)
        # shift and scale to [0, 1] range
        return (X - self.min_) / denom

    def fit_transform(self, X):
        # fit and transform in one step
        return self.fit(X).transform(X)


def train_test_split(X, y, test_size=0.2, random_state=None):
    """Randomly splits data into training and test sets."""
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    n = len(X)
    # figure out how many samples go in the test set
    n_test = int(n * test_size)
    # shuffle all indices so the split is random
    indices = rng.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]