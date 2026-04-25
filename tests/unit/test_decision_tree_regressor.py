import numpy as np
from rice_ml.supervised_learning.decision_tree_regressor import DecisionTreeRegressor


def test_perfect_fit():
    X = np.array([[1.0], [2.0], [3.0], [4.0],
                  [5.0], [6.0], [7.0], [8.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    model = DecisionTreeRegressor(max_depth=10).fit(X, y)
    assert np.allclose(model.predict(X), y, atol=1e-6)


def test_predict_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4))
    y = rng.standard_normal(30)
    model = DecisionTreeRegressor(max_depth=3).fit(X, y)
    assert model.predict(X).shape == (30,)


def test_r2_score():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    y = rng.standard_normal(50)
    model = DecisionTreeRegressor(max_depth=5).fit(X, y)
    assert model.score(X, y) > 0.5