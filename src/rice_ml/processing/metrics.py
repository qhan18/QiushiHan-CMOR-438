"""Evaluation metrics implemented from scratch."""
import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Compute classification accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.

    Returns
    -------
    accuracy : float
        Fraction of correctly classified samples.

    Example
    -------
    >>> accuracy_score([0, 1, 1, 0], [0, 1, 0, 0])
    0.75
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # just check how many predictions match the true labels
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true, y_pred):
    """
    Compute mean squared error regression loss.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    mse : float
        Average squared difference between predicted and true values.

    Example
    -------
    >>> mean_squared_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    0.0
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # square the errors so negatives dont cancel out positives
    errors = y_true - y_pred
    return np.mean(errors ** 2)


def r2_score(y_true, y_pred):
    """
    Compute R squared coefficient of determination.

    Measures how much variance in the target the model explains.
    A score of 1.0 means perfect prediction, 0.0 means the model
    does no better than predicting the mean, and negative values
    mean the model is worse than predicting the mean.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    r2 : float
        R squared score between negative infinity and 1.0.

    Example
    -------
    >>> r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    1.0
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # residual sum of squares, how wrong our predictions are
    ss_res = np.sum((y_true - y_pred) ** 2)
    # total sum of squares, how wrong a mean predictor would be
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # r2 of 1.0 means perfect, 0 means no better than predicting the mean
    return 1 - ss_res / ss_tot


def mean_absolute_error(y_true, y_pred):
    """
    Compute mean absolute error regression loss.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    mae : float
        Average absolute difference between predicted and true values.

    Example
    -------
    >>> mean_absolute_error([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    1.0
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # take absolute value so positive and negative errors dont cancel
    return np.mean(np.abs(y_true - y_pred))


def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix for classification results.

    Rows represent true labels and columns represent predicted labels.
    Entry [i, j] is the number of samples with true label i that were
    predicted as label j.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.

    Returns
    -------
    cm : np.ndarray of shape (n_classes, n_classes)
        Confusion matrix.

    Example
    -------
    >>> confusion_matrix([0, 1, 0, 1], [0, 1, 1, 0])
    array([[1, 1],
           [1, 1]])
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # figure out all unique classes across both arrays
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    # map class labels to matrix indices
    class_map = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=int)
    # fill in the matrix by counting each (true, predicted) pair
    for true_label, pred_label in zip(y_true, y_pred):
        cm[class_map[true_label], class_map[pred_label]] += 1
    return cm


def precision_score(y_true, y_pred):
    """
    Compute precision for binary classification.

    Precision is the fraction of positive predictions that were correct.
    A high precision means few false positives.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (0 or 1).

    Returns
    -------
    precision : float
        Fraction of positive predictions that were actually positive.
    """
    cm = confusion_matrix(y_true, y_pred)
    # true positives are correct positive predictions
    tp = cm[1, 1]
    # false positives are negative samples predicted as positive
    fp = cm[0, 1]
    # avoid division by zero if no positive predictions were made
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true, y_pred):
    """
    Compute recall for binary classification.

    Recall is the fraction of actual positives that were correctly identified.
    A high recall means few false negatives.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (0 or 1).

    Returns
    -------
    recall : float
        Fraction of actual positives that were correctly identified.
    """
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    # false negatives are positive samples we missed
    fn = cm[1, 0]
    # avoid division by zero if there were no actual positives
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0