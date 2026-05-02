"""K-Nearest Neighbors implemented from scratch."""
import numpy as np


class KNN:
    """
    K-Nearest Neighbors classifier using Euclidean distance.

    Makes predictions by finding the k closest training points to each
    test point and returning the majority class among those neighbors.
    Has no training phase since it memorizes the entire training set
    and does all computation at prediction time.

    Parameters
    ----------
    k : int
        How many neighbors to look at when making a prediction.

    Attributes
    ----------
    X_train : np.ndarray of shape (n_samples, n_features)
        Stored training feature matrix.
    y_train : np.ndarray of shape (n_samples,)
        Stored training labels.

    Example
    -------
    >>> model = KNN(k=5)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> print(model.score(X_test, y_test))
    """

    def __init__(self, k=3):
        self.k = k          # how many neighbors to look at
        self.X_train = None # store training data during fit
        self.y_train = None # store training labels during fit

    def fit(self, X, y):
        """
        Store training data for use during prediction.

        KNN doesnt actually learn anything, just memorizes
        the training data and uses it at prediction time.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Training labels.

        Returns
        -------
        self : KNN
            Fitted model instance.
        """
        # KNN doesnt actually learn anything, just memorizes the training data
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def _euclidean_distance(self, x1, x2):
        """
        Compute Euclidean distance between two points.

        Parameters
        ----------
        x1 : np.ndarray
            First point.
        x2 : np.ndarray
            Second point.

        Returns
        -------
        float
            Straight line distance between x1 and x2.
        """
        # standard straight line distance between two points
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X = np.asarray(X, dtype=float)
        # run prediction for each test point one at a time
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        """
        Predict the class label for a single sample.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            A single input sample.

        Returns
        -------
        label : int or str
            Predicted class label based on majority vote.
        """
        # figure out how far this point is from every training point
        distances = [self._euclidean_distance(x, x_train)
                     for x_train in self.X_train]
        # sort and grab the k closest ones
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        # whichever class shows up most among the neighbors wins
        values, counts = np.unique(k_labels, return_counts=True)
        return values[np.argmax(counts)]

    def score(self, X, y):
        """
        Compute classification accuracy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
        y : array-like of shape (n_samples,)
            True class labels.

        Returns
        -------
        accuracy : float
            Fraction of correctly classified samples.
        """
        return np.mean(self.predict(X) == np.asarray(y))