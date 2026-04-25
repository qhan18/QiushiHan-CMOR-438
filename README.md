# CMOR 438 / INDE 577: Data Science and Machine Learning

## Qiushi Han

## Overview

This repository contains a custom machine learning package developed for **CMOR 438 / INDE 577**.
The project implements classic **supervised and unsupervised learning algorithms from scratch**
using NumPy, organized into a clean and modular Python package called **`rice_ml`**.

The package is paired with structured **Jupyter notebooks** that demonstrate each algorithm
on real and synthetic datasets, emphasizing **mathematical intuition, algorithmic transparency,
and interpretability** rather than black-box usage.

---

## Project Highlights

- Fully custom implementations of core machine learning algorithms
- A well-structured, installable Python package (`rice_ml`)
- Separate modules for supervised learning, unsupervised learning, and preprocessing
- Educational notebooks demonstrating each algorithm step-by-step
- A comprehensive **pytest test suite** covering all major components

---

## Capabilities

### Supervised Learning
Implemented in `src/rice_ml/supervised_learning`:
- Linear Regression
- Logistic Regression
- k-Nearest Neighbors (KNN)
- Perceptron
- Multilayer Perceptron (Neural Network)
- Decision Trees
- Regression Trees
- Ensemble Methods

### Unsupervised Learning
Implemented in `src/rice_ml/unsupervised_learning`:
- K-Means Clustering
- DBSCAN
- Principal Component Analysis (PCA)
- Community Detection

### Data Processing Utilities
Implemented in `src/rice_ml/processing`:
- Feature standardization
- Train/test splitting
- Evaluation metrics

---

## Repository Structure
.
├── examples/
│   ├── Supervised_Learning/
│   └── Unsupervised_Learning/
├── src/
│   └── rice_ml/
│       ├── processing/
│       ├── supervised_learning/
│       ├── unsupervised_learning/
│       └── init.py
├── tests/unit/
├── README.md
├── requirements.txt
├── LICENSE
└── pyproject.toml

---

## Installation

```bash
git clone https://github.com/<your-username>/QiushiHan-CMOR-438.git
cd QiushiHan-CMOR-438
pip install -e .
```

## Example Usage

```python
from rice_ml.supervised_learning.linear_regression import LinearRegression
from rice_ml.unsupervised_learning.k_means import KMeans
from rice_ml.processing.preprocessing import standardize
```

## Running Tests

```bash
pytest
```

---

## Author and License

**Author:** Qiushi Han — Rice University, CMOR 438 / INDE 577, Spring 2026
**License:** MIT — see LICENSE file.