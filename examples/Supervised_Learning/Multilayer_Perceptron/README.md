# Multilayer Perceptron (MLP)

## Overview
Demonstrates a **feedforward neural network** with one hidden layer, implemented from scratch using sigmoid activations and backpropagation.

## Datasets
- **Breast Cancer Wisconsin** — 569 samples, 30 features, binary classification
- **Moons Dataset** — synthetic non-linearly separable data for comparison vs Perceptron

## Results
| Hidden Size | Breast Cancer Accuracy |
|---|---|
| 2 | 94.74% |
| 4 | 94.74% |
| 8 | 95.61% |
| 16-64 | 94.74% |

| Model | Moons Accuracy |
|---|---|
| Perceptron | 90.00% |
| MLP | 86.67% |

## Key Concepts
- Forward pass, backpropagation, sigmoid activation, hidden layers, Universal Approximation Theorem, hyperparameter tuning