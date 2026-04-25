import numpy as np
from rice_ml.processing.metrics import (
    accuracy_score, mean_squared_error, r2_score,
    mean_absolute_error, confusion_matrix,
    precision_score, recall_score
)


def test_accuracy_score():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    assert np.isclose(accuracy_score(y_true, y_pred), 0.8)


def test_mean_squared_error():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert np.isclose(mean_squared_error(y_true, y_pred), 0.0)


def test_r2_score_perfect():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(r2_score(y_true, y_pred), 1.0)


def test_mean_absolute_error():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    assert np.isclose(mean_absolute_error(y_true, y_pred), 1.0)


def test_confusion_matrix_shape():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)


def test_precision_recall():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])
    assert precision_score(y_true, y_pred) == 1.0
    assert np.isclose(recall_score(y_true, y_pred), 2/3, atol=1e-6)