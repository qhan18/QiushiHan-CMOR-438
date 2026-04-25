import numpy as np
from rice_ml.unsupervised_learning.k_means import KMeans


def test_cluster_count():
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.standard_normal((50, 2)) + np.array([5, 5]),
        rng.standard_normal((50, 2)) + np.array([-5, -5]),
        rng.standard_normal((50, 2)) + np.array([5, -5]),
    ])
    model = KMeans(k=3).fit(X)
    assert len(np.unique(model.labels_)) == 3


def test_predict_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))
    model = KMeans(k=4).fit(X)
    assert model.predict(X).shape == (100,)


def test_centroid_count():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 3))
    model = KMeans(k=5).fit(X)
    assert model.centroids.shape == (5, 3)