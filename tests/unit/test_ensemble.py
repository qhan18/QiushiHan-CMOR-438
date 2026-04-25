import numpy as np
from rice_ml.supervised_learning.ensemble import RandomForestClassifier


def test_predict_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    y = (rng.standard_normal(50) > 0).astype(int)
    model = RandomForestClassifier(n_estimators=5, max_depth=3).fit(X, y)
    assert model.predict(X).shape == (50,)


def test_predict_binary():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    y = (rng.standard_normal(50) > 0).astype(int)
    model = RandomForestClassifier(n_estimators=5, max_depth=3).fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_better_than_chance():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))
    y = (rng.standard_normal(100) > 0).astype(int)
    model = RandomForestClassifier(n_estimators=10, max_depth=5).fit(X, y)
    assert model.score(X, y) > 0.5