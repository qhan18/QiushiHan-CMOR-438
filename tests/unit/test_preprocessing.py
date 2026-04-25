import numpy as np
from rice_ml.processing.preprocessing import (
    StandardScaler, MinMaxScaler, train_test_split
)


def test_standard_scaler_mean_std():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    assert np.isclose(X_scaled.mean(axis=0), 0, atol=1e-8).all()
    assert np.isclose(X_scaled.std(axis=0), 1, atol=1e-8).all()


def test_minmax_scaler_range():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = MinMaxScaler().fit(X)
    X_scaled = scaler.transform(X)
    assert np.isclose(X_scaled.min(), 0.0, atol=1e-8)
    assert np.isclose(X_scaled.max(), 1.0, atol=1e-8)


def test_train_test_split_sizes():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    assert len(X_train) == 40
    assert len(X_test) == 10