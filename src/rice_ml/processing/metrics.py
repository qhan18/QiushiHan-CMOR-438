"""Evaluation metrics implemented from scratch."""
import numpy as np


def accuracy_score(y_true, y_pred):
    """Fraction of correctly classified samples."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # just check how many predictions match the true labels
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true, y_pred):
    """Mean squared error regression loss."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # square the errors so negatives don't cancel out positives
    errors = y_true - y_pred
    return np.mean(errors ** 2)


def r2_score(y_true, y_pred):
    """R squared coefficient of determination, measures how much variance the model explains."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # residual sum of squares, how wrong our predictions are
    ss_res = np.sum((y_true - y_pred) ** 2)
    # total sum of squares, how wrong a mean predictor would be
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # r2 of 1.0 means perfect, 0 means no better than predicting the mean
    return 1 - ss_res / ss_tot


def mean_absolute_error(y_true, y_pred):
    """Mean absolute error, average magnitude of errors."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # take absolute value so positive and negative errors don't cancel
    return np.mean(np.abs(y_true - y_pred))


def confusion_matrix(y_true, y_pred):
    """Compute confusion matrix for binary classification."""
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
    """Precision, of all predicted positives how many were actually positive."""
    cm = confusion_matrix(y_true, y_pred)
    # true positives are correct positive predictions
    tp = cm[1, 1]
    # false positives are negative samples predicted as positive
    fp = cm[0, 1]
    # avoid division by zero if no positive predictions were made
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true, y_pred):
    """Recall, of all actual positives how many did we correctly find."""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    # false negatives are positive samples we missed
    fn = cm[1, 0]
    # avoid division by zero if there were no actual positives
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0