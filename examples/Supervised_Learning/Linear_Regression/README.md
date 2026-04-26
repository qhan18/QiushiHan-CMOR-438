# Linear Regression

## Overview
This notebook demonstrates **Linear Regression** implemented from scratch in the `rice_ml` package, using three different methods:

1. **OLS** (Ordinary Least Squares) — closed-form normal equation solution
2. **Ridge Regression** — closed-form with L2 regularization penalty
3. **Gradient Descent** — iterative weight update optimization

## Dataset
**Diabetes Dataset** (sklearn) — 442 patients, 10 physiological features, continuous target representing disease progression.

## Results
| Method | R² Score |
|---|---|
| OLS | 0.4526 |
| Ridge (α=10) | 0.4572 |
| Gradient Descent | 0.4555 |

## Key Concepts
- Normal equation, gradient descent, L2 regularization, bias-variance tradeoff
