# Ensemble Methods (Random Forest)

## Overview
Demonstrates a **Random Forest Classifier** implemented from scratch using bagging and feature randomness to combine multiple decision trees.

## Dataset
**Wine Dataset** (sklearn) — 178 samples, 13 chemical features, 3 wine classes.

## Results
| Model | Accuracy |
|---|---|
| Single Decision Tree (depth 5) | 97.22% |
| Random Forest (50 trees, depth 5) | 100.00% |

Number of trees vs accuracy
| Trees | Accuracy |
|---|---|
| 1 | 77.78% |
| 5 | 86.11% |
| 10 | 91.67% |
| 20 | 94.44% |
| 50 | 100.00% |
| 100 | 97.22% |

## Key Concepts
- Bagging, bootstrap sampling, feature randomness, variance reduction, ensemble voting