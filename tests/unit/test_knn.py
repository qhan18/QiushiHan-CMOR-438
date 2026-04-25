import numpy as np
from rice_ml.supervised_learning.knn import KNN


def test_perfect_classification():
    X_train = np.array([[1.0, 1.0], [2.0, 2.0],
                        [-1.0, -1.0], [-2.0, -2.0]])
    y_train = np.array([1, 1, 0, 0])
    model = KNN(k=1).fit(X_train, y_train)
    assert model.score(X_train, y_train) == 1.0


def test_predict_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    y = (rng.standard_normal(30) > 0).astype(int)
    model = KNN(k=3).fit(X, y)
    assert model.predict(X).shape == (30,)


def test_k1_memorizes():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([10, 20, 30, 40])
    model = KNN(k=1).fit(X, y)
    assert list(model.predict(X)) == [10, 20, 30, 40]