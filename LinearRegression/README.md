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

Linear Regression models the relationship between input features \(X\) and target values \(y\) as:

\[
\hat{y} = Xw + b
\]

The objective is to minimize Mean Squared Error:

\[
J(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

### Gradient Descent
Gradient descent iteratively updates the model parameters:

\[
w \leftarrow w - \eta \frac{\partial J}{\partial w}
\]

### Closed-Form Solution
The analytical least-squares solution is:

\[
\theta = (X^T X)^{-1} X^T y
\]

## Why Two Implementations?

The Gradient Descent version helps demonstrate optimization in machine learning, while the Closed-Form solution shows the direct linear algebra approach. Together, they make it easier to compare tradeoffs between iterative and analytical methods.

## Project Structure

