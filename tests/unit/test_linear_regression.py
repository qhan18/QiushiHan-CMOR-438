import numpy as np
from rice_ml.supervised_learning.linear_regression import LinearRegression


def test_perfect_fit():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])  # y = 2x + 1
    model = LinearRegression().fit(X, y)
    assert np.isclose(model.coef_[0], 2.0, atol=1e-8)
    assert np.isclose(model.intercept_, 1.0, atol=1e-8)


def test_predict_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3))
    y = rng.standard_normal(20)
    model = LinearRegression().fit(X, y)
    assert model.predict(X).shape == (20,)


def test_r2_perfect():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])
    model = LinearRegression().fit(X, y)
    assert np.isclose(model.score(X, y), 1.0, atol=1e-8)