import numpy as np
from rice_ml.unsupervised_learning.pca import PCA


def test_output_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 10))
    pca = PCA(n_components=3).fit(X)
    assert pca.transform(X).shape == (100, 3)


def test_explained_variance_ratio_sums_to_one():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 5))
    pca = PCA(n_components=5).fit(X)
    assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0, atol=1e-6)


def test_fit_transform_matches():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 6))
    pca = PCA(n_components=2)
    result1 = pca.fit_transform(X)
    result2 = pca.transform(X)
    assert np.allclose(result1, result2)