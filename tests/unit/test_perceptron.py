import numpy as np
from rice_ml.supervised_learning.perceptron import Perceptron


def test_linearly_separable():
    X = np.array([[1.0, 1.0], [2.0, 2.0],
                  [-1.0, -1.0], [-2.0, -2.0]])
    y = np.array([1, 1, 0, 0])
    model = Perceptron(learning_rate=0.1, max_iter=1000).fit(X, y)
    assert model.score(X, y) == 1.0


def test_predict_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3))
    y = (rng.standard_normal(20) > 0).astype(int)
    model = Perceptron().fit(X, y)
    assert model.predict(X).shape == (20,)


def test_predict_binary():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 2))
    y = (rng.standard_normal(20) > 0).astype(int)
    model = Perceptron().fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})