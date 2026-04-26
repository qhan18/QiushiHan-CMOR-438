import numpy as np
from rice_ml.supervised_learning.multilayer_perceptron import MLP


def test_predict_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    y = (rng.standard_normal(50) > 0).astype(int)
    model = MLP(hidden_size=8, max_iter=100).fit(X, y)
    assert model.predict(X).shape == (50,)


def test_predict_proba_range():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    y = (rng.standard_normal(30) > 0).astype(int)
    model = MLP(hidden_size=8, max_iter=100).fit(X, y)
    proba = model.predict_proba(X)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_predict_binary():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    y = (rng.standard_normal(30) > 0).astype(int)
    model = MLP(hidden_size=8, max_iter=100).fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})