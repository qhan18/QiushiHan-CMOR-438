# Decision Tree Classifier

## Overview
Demonstrates a **Decision Tree Classifier** implemented from scratch using entropy-based information gain for split selection.

## Dataset
**Wine Dataset** (sklearn) — 178 samples, 13 chemical features, 3 wine cultivar classes.

## Results
| Max Depth | Train Accuracy | Test Accuracy |
|---|---|---|
| 1 | 59.86% | 61.11% |
| 2 | 95.77% | 100.00% |
| 3 | 99.30% | 100.00% |
| 5 | 100.00% | 97.22% |
| 7+ | 100.00% | 97.22% |

## Key Concepts
- Entropy, information gain, recursive splitting, axis-aligned boundaries, overfitting, optimal depth selection