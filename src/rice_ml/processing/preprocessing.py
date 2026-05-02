"""Preprocessing utilities implemented from scratch."""
import numpy as np


class StandardScaler:
    """
    Standardize features to have zero mean and unit variance.

    Computes the mean and standard deviation from training data during
    fit, then uses those values to transform any dataset. Always fit
    on training data only and use transform on both train and test sets
    to avoid data leakage.

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_features,)
        Mean of each feature computed during fit.
    std_ : np.ndarray of shape (n_features,)
        Standard deviation of each feature computed during fit.

    Example
    -------
    >>> scaler = StandardScaler()
    >>> X_train_scaled = scaler.fit_transform(X_train)
    >>> X_test_scaled = scaler.transform(X_test)
    """

    def __init__(self):
        self.mean_ = None  # will store the mean of each feature after fitting
        self.std_ = None   # will store the std of each feature after fitting

    def fit(self, X):
        """
        Compute mean and standard deviation from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data to compute statistics from.

        Returns
        -------
        self : StandardScaler
            Fitted scaler instance.
        """
        X = np.asarray(X, dtype=float)
        # compute mean and std across all training samples for each feature
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # if a feature has zero variance, set std to 1 so we dont divide by zero
        self.std_ = np.where(self.std_ == 0, 1, self.std_)
        return self

    def transform(self, X):
        """
        Apply standardization to input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to standardize using fitted mean and std.

        Returns
        -------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Standardized data with zero mean and unit variance.
        """
        X = np.asarray(X, dtype=float)
        # apply z-score normalization using the stored mean and std
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """
        Fit to data and transform it in one step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data to fit and transform.

        Returns
        -------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Standardized training data.
        """
        # fit and transform in one step, useful for training data
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """
        Reverse standardization to recover original feature values.

        Parameters
        ----------
        X_scaled : array-like of shape (n_samples, n_features)
            Standardized data to reverse transform.

        Returns
        -------
        X_original : np.ndarray of shape (n_samples, n_features)
            Data in the original feature space.
        """
        # undo the scaling to get back to the original feature space
        return X_scaled * self.std_ + self.mean_


class MinMaxScaler:
    """
    Scale features to the range [0, 1].

    Computes the minimum and maximum from training data during fit,
    then scales any dataset to the [0, 1] range using those values.
    Always fit on training data only to avoid data leakage.

    Attributes
    ----------
    min_ : np.ndarray of shape (n_features,)
        Minimum value of each feature computed during fit.
    max_ : np.ndarray of shape (n_features,)
        Maximum value of each feature computed during fit.

    Example
    -------
    >>> scaler = MinMaxScaler()
    >>> X_train_scaled = scaler.fit_transform(X_train)
    >>> X_test_scaled = scaler.transform(X_test)
    """

    def __init__(self):
        self.min_ = None  # minimum value of each feature
        self.max_ = None  # maximum value of each feature

    def fit(self, X):
        """
        Compute minimum and maximum from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data to compute statistics from.

        Returns
        -------
        self : MinMaxScaler
            Fitted scaler instance.
        """
        X = np.asarray(X, dtype=float)
        # store min and max so we can scale new data consistently
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        """
        Scale input data to [0, 1] range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to scale using fitted min and max.

        Returns
        -------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Data scaled to [0, 1] range.
        """
        X = np.asarray(X, dtype=float)
        # handle edge case where a feature is constant (max == min)
        denom = np.where(self.max_ - self.min_ == 0, 1, self.max_ - self.min_)
        # shift and scale to [0, 1] range
        return (X - self.min_) / denom

    def fit_transform(self, X):
        """
        Fit to data and transform it in one step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data to fit and transform.

        Returns
        -------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Data scaled to [0, 1] range.
        """
        # fit and transform in one step
        return self.fit(X).transform(X)


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Randomly split data into training and test sets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix to split.
    y : array-like of shape (n_samples,)
        Target array to split.
    test_size : float
        Fraction of samples to use for the test set.
        Must be between 0 and 1.
    random_state : int or None
        Seed for the random number generator. Set for reproducibility.

    Returns
    -------
    X_train : np.ndarray
        Training feature matrix.
    X_test : np.ndarray
        Test feature matrix.
    y_train : np.ndarray
        Training target array.
    y_test : np.ndarray
        Test target array.

    Example
    -------
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.2, random_state=42)
    """
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