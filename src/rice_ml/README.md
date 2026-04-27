# rice_ml

A from-scratch machine learning library implemented in Python using only NumPy.
Built for CMOR 438 / INDE 577 at Rice University by Qiushi Han.

**Author:** Qiushi Han
**Email:** qh23@rice.edu
**GitHub:** github.com/qhan18

---

## Installation

From the root of the repository:

```bash
pip install -e .
```

---

## Package Structure
rice_ml/
├── supervised_learning/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── knn.py
│   ├── perceptron.py
│   ├── multilayer_perceptron.py
│   ├── decision_tree_classifier.py
│   ├── decision_tree_regressor.py
│   └── ensemble.py
├── unsupervised_learning/
│   ├── k_means.py
│   ├── dbscan.py
│   ├── pca.py
│   └── label_propagation.py
└── processing/
├── preprocessing.py
└── metrics.py

---

## Quick Reference

```python
# Supervised Learning
from rice_ml.supervised_learning.linear_regression import LinearRegression
from rice_ml.supervised_learning.logistic_regression import LogisticRegression
from rice_ml.supervised_learning.knn import KNN
from rice_ml.supervised_learning.perceptron import Perceptron
from rice_ml.supervised_learning.multilayer_perceptron import MLP
from rice_ml.supervised_learning.decision_tree_classifier import DecisionTreeClassifier
from rice_ml.supervised_learning.decision_tree_regressor import DecisionTreeRegressor
from rice_ml.supervised_learning.ensemble import RandomForestClassifier

# Unsupervised Learning
from rice_ml.unsupervised_learning.k_means import KMeans
from rice_ml.unsupervised_learning.dbscan import DBSCAN
from rice_ml.unsupervised_learning.pca import PCA
from rice_ml.unsupervised_learning.label_propagation import LabelPropagation

# Processing
from rice_ml.processing.preprocessing import StandardScaler, MinMaxScaler, train_test_split
from rice_ml.processing.metrics import accuracy_score, mean_squared_error, r2_score
```

---

## API Convention

All supervised learning models follow the sklearn-style API:

```python
model = Algorithm(hyperparameters)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = model.score(X_test, y_test)
```

All unsupervised learning models follow:

```python
model = Algorithm(hyperparameters)
model.fit(X)
labels = model.labels_
```

---

## Dependencies

- numpy
- No other dependencies for core algorithms
- sklearn used only for dataset loading in example notebooks