import numpy as np
from rice_ml.supervised_learning.decision_tree_classifier import DecisionTreeClassifier


def test_perfect_fit():
    X = np.array([[1.0], [2.0], [3.0], [4.0],
                  [5.0], [6.0], [7.0], [8.0]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert model.score(X, y) == 1.0


def test_predict_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4))
    y = (rng.standard_normal(30) > 0).astype(int)
    model = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert model.predict(X).shape == (30,)


def test_max_depth_respected():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    y = (rng.standard_normal(50) > 0).astype(int)
    model = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert model.root.left.is_leaf() and model.root.right.is_leaf()