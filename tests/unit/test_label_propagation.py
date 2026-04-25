import numpy as np
from rice_ml.unsupervised_learning.label_propagation import LabelPropagation


def test_labeled_points_correct():
    X = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.0],
                  [10.0, 10.0], [10.1, 10.1], [10.2, 10.0]])
    y = np.array([0, -1, -1, 1, -1, -1])
    model = LabelPropagation(gamma=1).fit(X, y)
    assert model.labels_[0] == 0
    assert model.labels_[3] == 1


def test_output_shape():
    X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0],
                  [5.0, 5.0], [5.5, 5.5], [6.0, 5.0]])
    y = np.array([0, -1, -1, 1, -1, -1])
    model = LabelPropagation(gamma=1)
    labels = model.fit_predict(X, y)
    assert labels.shape == (6,)


def test_propagates_to_unlabeled():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0],
                  [10.0, 0.0], [10.1, 0.0], [10.2, 0.0]])
    y = np.array([0, -1, -1, 1, -1, -1])
    model = LabelPropagation(gamma=1).fit(X, y)
    assert model.labels_[1] == 0
    assert model.labels_[4] == 1