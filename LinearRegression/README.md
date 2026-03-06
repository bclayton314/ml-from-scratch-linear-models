# Linear Regression from Scratch (NumPy)

This repo implements linear regression (least squares) from scratch with:
- Closed-form solver (pseudoinverse; optional normal equation)
- Gradient descent solver (batch + mini-batch)
- Ridge regression (L2) closed-form
- K-fold cross-validation + metrics

## Why
Goal: connect linear algebra (subspaces, rank, projections) to a working ML implementation.

## Quickstart
```bash
pip install -e .
python experiments/synthetic_1d.py
python experiments/collinearity_rank_deficiency.py

