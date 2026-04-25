import numpy as np
from rice_ml.unsupervised_learning.dbscan import DBSCAN


def test_finds_clusters():
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.standard_normal((50, 2)) * 0.3 + np.array([0, 0]),
        rng.standard_normal((50, 2)) * 0.3 + np.array([10, 10]),
    ])
    model = DBSCAN(eps=1.0, min_samples=3).fit(X)
    unique_labels = np.unique(model.labels_[model.labels_ != -1])
    assert len(unique_labels) >= 2


def test_noise_label():
    X = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2],
                  [100.0, 100.0]])  # last point is noise
    model = DBSCAN(eps=0.5, min_samples=2).fit(X)
    assert model.labels_[3] == -1


def test_fit_predict_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 2))
    model = DBSCAN(eps=0.5, min_samples=5)
    labels = model.fit_predict(X)
    assert labels.shape == (80,)