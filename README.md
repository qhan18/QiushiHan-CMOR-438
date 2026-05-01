# CMOR 438 / INDE 577: Data Science and Machine Learning

## Qiushi Han | Rice University | Spring 2026
**Instructor:** Dr. Randy R. Davila

[![CI](https://github.com/qhan18/QiushiHan-CMOR-438/actions/workflows/ci.yml/badge.svg)](https://github.com/qhan18/QiushiHan-CMOR-438/actions/workflows/ci.yml)

---

## Overview

This repository contains a custom machine learning package developed for **CMOR 438 / INDE 577: Data Science and Machine Learning**. The project implements classic supervised and unsupervised learning algorithms entirely from scratch using NumPy, organized into a clean and modular Python package called **`rice_ml`**.

The package is paired with structured Jupyter notebooks that demonstrate each algorithm on real and synthetic datasets, emphasizing mathematical intuition, algorithmic transparency, and interpretability rather than black-box usage.

---

## Project Highlights

- 14 machine learning algorithms implemented from scratch using only NumPy
- A well-structured, installable Python package (`rice_ml`) using modern `pyproject.toml` build system
- Separate modules for supervised learning, unsupervised learning, and data processing utilities
- 12 educational Jupyter notebooks demonstrating each algorithm with real datasets
- 45 pytest unit tests ensuring correctness and robustness
- GitHub Actions CI pipeline that automatically runs tests on every push

---

## Algorithms Implemented

### Supervised Learning
Implemented in `src/rice_ml/supervised_learning/`

| Algorithm | File | Key Concepts |
|---|---|---|
| Linear Regression | [`linear_regression.py`](src/rice_ml/supervised_learning/linear_regression.py) | OLS, Ridge, Gradient Descent |
| Logistic Regression | [`logistic_regression.py`](src/rice_ml/supervised_learning/logistic_regression.py) | Sigmoid, Binary Cross-Entropy |
| K-Nearest Neighbors | [`knn.py`](src/rice_ml/supervised_learning/knn.py) | Euclidean Distance, Majority Vote |
| Perceptron | [`perceptron.py`](src/rice_ml/supervised_learning/perceptron.py) | Rosenblatt Learning Rule |
| Multilayer Perceptron | [`multilayer_perceptron.py`](src/rice_ml/supervised_learning/multilayer_perceptron.py) | Backpropagation, Sigmoid Activation |
| Decision Tree Classifier | [`decision_tree_classifier.py`](src/rice_ml/supervised_learning/decision_tree_classifier.py) | Entropy, Information Gain |
| Decision Tree Regressor | [`decision_tree_regressor.py`](src/rice_ml/supervised_learning/decision_tree_regressor.py) | Variance Reduction |
| Random Forest | [`ensemble.py`](src/rice_ml/supervised_learning/ensemble.py) | Bagging, Feature Randomness |

### Unsupervised Learning
Implemented in `src/rice_ml/unsupervised_learning/`

| Algorithm | File | Key Concepts |
|---|---|---|
| K-Means Clustering | [`k_means.py`](src/rice_ml/unsupervised_learning/k_means.py) | Lloyd's Algorithm, Inertia, Elbow Method |
| DBSCAN | [`dbscan.py`](src/rice_ml/unsupervised_learning/dbscan.py) | Density-Based, Noise Detection |
| PCA | [`pca.py`](src/rice_ml/unsupervised_learning/pca.py) | Eigendecomposition, Explained Variance |
| Label Propagation | [`label_propagation.py`](src/rice_ml/unsupervised_learning/label_propagation.py) | RBF Kernel, Semi-Supervised Learning |

### Data Processing Utilities
Implemented in `src/rice_ml/processing/`

| Utility | File | Key Concepts |
|---|---|---|
| Preprocessing | [`preprocessing.py`](src/rice_ml/processing/preprocessing.py) | StandardScaler, MinMaxScaler, Train/Test Split |
| Metrics | [`metrics.py`](src/rice_ml/processing/metrics.py) | Accuracy, MSE, R², Confusion Matrix, Precision, Recall |

---

## Repository Structure
QiushiHan-CMOR-438/
├── .github/workflows/ci.yml
├── examples/
│   ├── Supervised_Learning/
│   │   ├── Linear_Regression/
│   │   ├── Logistic_Regression/
│   │   ├── K_Nearest_Neighbors/
│   │   ├── Perceptron/
│   │   ├── Multilayer_Perceptron/
│   │   ├── Decision_Trees/
│   │   ├── Regression_Trees/
│   │   └── Ensemble_Methods/
│   └── Unsupervised_Learning/
│       ├── K_Means_Clustering/
│       ├── DBSCAN/
│       ├── PCA/
│       └── Community_Detection/
├── src/rice_ml/
│   ├── supervised_learning/
│   ├── unsupervised_learning/
│   └── processing/
├── tests/unit/
├── README.md
├── LICENSE
├── pyproject.toml
└── requirements.txt

---

## Installation

```bash
git clone https://github.com/qhan18/QiushiHan-CMOR-438.git
cd QiushiHan-CMOR-438
pip install -e .
```

## Example Usage

```python
from rice_ml.supervised_learning.linear_regression import LinearRegression
from rice_ml.supervised_learning.knn import KNN
from rice_ml.unsupervised_learning.k_means import KMeans
from rice_ml.unsupervised_learning.pca import PCA
from rice_ml.processing.preprocessing import StandardScaler, train_test_split
from rice_ml.processing.metrics import accuracy_score, r2_score
```

## Running Tests

```bash
pytest
```

All 45 unit tests pass across 14 algorithm implementations.

---

## Demo Notebooks

Each algorithm has a dedicated Jupyter notebook in the `examples/` folder demonstrating real dataset usage, visualizations, and analysis.

| Notebook | Dataset | Key Result |
|---|---|---|
| [Linear Regression](examples/Supervised_Learning/Linear_Regression/) | Diabetes | R² = 0.45 (OLS), Ridge, GD comparison |
| [Logistic Regression](examples/Supervised_Learning/Logistic_Regression/) | Breast Cancer | 98.2% accuracy |
| [KNN](examples/Supervised_Learning/K_Nearest_Neighbors/) | Wine | 100% accuracy at k=7 |
| [Perceptron](examples/Supervised_Learning/Perceptron/) | Breast Cancer | 94.74% accuracy |
| [MLP](examples/Supervised_Learning/Multilayer_Perceptron/) | Breast Cancer | 95.61% accuracy, moons comparison |
| [Decision Tree Classifier](examples/Supervised_Learning/Decision_Trees/) | Wine | 97.22% accuracy, depth vs overfitting |
| [Decision Tree Regressor](examples/Supervised_Learning/Regression_Trees/) | Diabetes | R² analysis, step-function visualization |
| [Random Forest](examples/Supervised_Learning/Ensemble_Methods/) | Wine | 100% accuracy, trees vs accuracy curve |
| [K-Means](examples/Unsupervised_Learning/K_Means_Clustering/) | Synthetic Blobs | 4 clusters recovered, elbow method |
| [DBSCAN](examples/Unsupervised_Learning/DBSCAN/) | Moons, Noisy Blobs | Arbitrary shape detection, noise handling |
| [PCA](examples/Unsupervised_Learning/PCA/) | Digits | 40 components for 95% variance, reconstruction |
| [Label Propagation](examples/Unsupervised_Learning/Community_Detection/) | Blobs, Moons | 87.5% accuracy with 20% labeled data |

---

## Author and License

**Author:** Qiushi Han
**Email:** qh23@rice.edu
**GitHub:** github.com/qhan18
**Institution:** Rice University
**Course:** CMOR 438 / INDE 577, Spring 2026
**Instructor:** Dr. Randy R. Davila
**License:** MIT, see LICENSE file