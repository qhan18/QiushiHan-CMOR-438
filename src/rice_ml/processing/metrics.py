"""Evaluation metrics implemented from scratch."""
import numpy as np


def accuracy_score(y_true, y_pred):
    """Fraction of correctly classified samples."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true, y_pred):
    """Mean squared error regression loss."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    """R² coefficient of determination."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def mean_absolute_error(y_true, y_pred):
    """Mean absolute error regression loss."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))


def confusion_matrix(y_true, y_pred):
    """Compute confusion matrix for binary classification."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    class_map = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((n, n), dtype=int)
    # count predictions vs true labels
    for t, p in zip(y_true, y_pred):
        matrix[class_map[t], class_map[p]] += 1
    return matrix


def precision_score(y_true, y_pred):
    """Precision for binary classification."""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true, y_pred):
    """Recall for binary classification."""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fn = cm[1, 0]
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0