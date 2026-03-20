# Linear Regression From Scratch in Python

This project implements Linear Regression from scratch in Python using two approaches:

1. **Gradient Descent**
2. **Closed-Form Solution (Normal Equation)**

The purpose of this project is to understand the mathematics and implementation details of linear regression without using machine learning libraries such as scikit-learn.

## Features

- Linear Regression built from scratch with NumPy
- Gradient Descent implementation
- Closed-Form / Normal Equation implementation
- Prediction support
- Mean Squared Error (MSE)
- R² score
- Data visualization with Matplotlib

## Mathematical Background

Linear Regression models the relationship between input features X and target values y as:

**ŷ = wx + b**

Where:

- x = input
- w = weight (slope)
- b = bias (intercept)
- ŷ = predicted value

For multiple features:

**ŷ = Xw + b**

Or, using an intercept column:

**ŷ = Xθ**


### Gradient Descent
Instead of solving analytically, we minimize the loss using gradient descent.

**w = w - η * ∂J/∂w**

**b = b - η * ∂J/∂b**


### Closed-Form Solution (Normal Equation)
Instead of iterative updates, we solve for 𝜃 directly:

**θ = (XᵀX)⁻¹ Xᵀy**

In practice, we often use the pseudo-inverse instead of matrix inversion:

**θ = (XᵀX)^(-1) Xᵀy**   ← theoretical

__θ = pinv(X)*y__        ← practical


### Loss Function (Mean Squared Error)
The objective is to minimize Mean Squared Error:

**J(w, b) = (1/n) * Σ (yᵢ - ŷᵢ)²**

Substituting the model:

**J(w, b) = (1/n) * Σ (yᵢ - (wxᵢ + b))²**

Key properties:

- Penalizes large errors more heavily

- Convex → guarantees a global minimum





## Why Two Implementations?

The Gradient Descent version helps demonstrate optimization in machine learning, while the Closed-Form solution shows the direct linear algebra approach. Together, they make it easier to compare tradeoffs between iterative and analytical methods.

## Project Structure




