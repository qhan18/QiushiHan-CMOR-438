# Decision Tree Regressor (Regression Trees)

## Overview
Demonstrates a **Decision Tree Regressor** implemented from scratch using variance reduction for split selection. Predicts continuous targets by averaging values at each leaf node.

## Dataset
**Diabetes Dataset** (sklearn) — 442 samples, 10 features, continuous target (disease progression).

## Results
| Max Depth | Train R² | Test R² |
|---|---|---|
| 1 | 0.3043 | 0.1385 |
| 2 | 0.4473 | 0.2703 |
| 3 | 0.5170 | 0.3899 |
| 5 | 0.6606 | 0.2721 |
| 10 | 0.9382 | 0.0843 |
| 15+ | 1.0000 | negative |

## Key Concepts
- Variance reduction, piecewise-constant predictions, step functions, overfitting, optimal depth selection