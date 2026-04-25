import numpy as np
from rice_ml.supervised_learning.logistic_regression import LogisticRegression


def test_perfect_classification():
    X = np.array([[1.0], [2.0], [3.0], [4.0],
                  [-1.0], [-2.0], [-3.0], [-4.0]])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    model = LogisticRegression(learning_rate=0.1, max_iter=1000).fit(X, y)
    assert model.score(X, y) == 1.0


def test_predict_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    y = (rng.standard_normal(50) > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    assert model.predict(X).shape == (50,)


def test_predict_proba_range():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 2))
    y = (rng.standard_normal(20) > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    proba = model.predict_proba(X)
    assert np.all(proba >= 0) and np.all(proba <= 1)