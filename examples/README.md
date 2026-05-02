# Examples

This folder contains Jupyter notebooks demonstrating each algorithm in the `rice_ml` package on real and synthetic datasets. Each notebook includes data exploration, preprocessing, modeling, evaluation, and a detailed discussion of results.

---

## Supervised Learning

| Algorithm | Dataset | Notebook |
|---|---|---|
| [Linear Regression](Supervised_Learning/Linear_Regression/) | Diabetes | OLS, Ridge, and Gradient Descent comparison |
| [Logistic Regression](Supervised_Learning/Logistic_Regression/) | Breast Cancer | 98.2% accuracy, confusion matrix, probability distribution |
| [K-Nearest Neighbors](Supervised_Learning/K_Nearest_Neighbors/) | Wine | 100% accuracy at k=7, decision boundary visualization |
| [Perceptron](Supervised_Learning/Perceptron/) | Breast Cancer | 94.74% accuracy, linear vs non-linear comparison |
| [Multilayer Perceptron](Supervised_Learning/Multilayer_Perceptron/) | Breast Cancer | 95.61% accuracy, hidden layer size sweep, moons comparison |
| [Decision Tree Classifier](Supervised_Learning/Decision_Trees/) | Wine | 97.22% accuracy, depth vs overfitting analysis |
| [Decision Tree Regressor](Supervised_Learning/Regression_Trees/) | Diabetes | R² analysis, step-function visualization |
| [Random Forest](Supervised_Learning/Ensemble_Methods/) | Wine | 100% accuracy, number of trees vs accuracy curve |

---

## Unsupervised Learning

| Algorithm | Dataset | Notebook |
|---|---|---|
| [K-Means Clustering](Unsupervised_Learning/K_Means_Clustering/) | Synthetic Blobs | 4 clusters recovered, elbow method for optimal k |
| [DBSCAN](Unsupervised_Learning/DBSCAN/) | Moons, Noisy Blobs | Arbitrary shape detection, noise handling, eps sensitivity |
| [PCA](Unsupervised_Learning/PCA/) | Digits | 40 components for 95% variance, digit reconstruction |
| [Label Propagation](Unsupervised_Learning/Community_Detection/) | Blobs, Moons | 87.5% accuracy with only 20% labeled data |

---

## Running the Notebooks

Make sure you have installed the package first:

```bash
pip install -e .
```

Then launch Jupyter:

```bash
jupyter notebook
```

Navigate to any subfolder and open the notebook to run it.