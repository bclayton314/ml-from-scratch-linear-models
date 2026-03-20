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




1. Introduction

Linear regression is one of the most fundamental algorithms in machine learning and statistics. It models the relationship between an input variable and a target variable by fitting a linear function to observed data.

This report explains:

the mathematical formulation of linear regression

the Mean Squared Error loss function

how gradient descent is used to learn model parameters

the geometric intuition behind optimization

2. The Linear Model

In its simplest form (single feature), linear regression assumes:

𝑦
^
=
𝑤
𝑥
+
𝑏
y
^
	​

=wx+b

Where:

𝑥
∈
𝑅
x∈R is the input

𝑤
∈
𝑅
w∈R is the slope (weight)

𝑏
∈
𝑅
b∈R is the intercept

𝑦
^
y
^
	​

 is the predicted value

For multiple features, this generalizes to:

𝑦
^
=
𝑋
𝑤
+
𝑏
y
^
	​

=Xw+b

or equivalently (with intercept absorbed into 
𝑋
X):

𝑦
^
=
𝑋
𝜃
y
^
	​

=Xθ
3. Objective Function: Mean Squared Error

To evaluate how well the model fits the data, we use the Mean Squared Error (MSE):

𝐽
(
𝑤
,
𝑏
)
=
1
𝑛
∑
𝑖
=
1
𝑛
(
𝑦
𝑖
−
𝑦
^
𝑖
)
2
J(w,b)=
n
1
	​

i=1
∑
n
	​

(y
i
	​

−
y
^
	​

i
	​

)
2

Substituting the model:

𝐽
(
𝑤
,
𝑏
)
=
1
𝑛
∑
𝑖
=
1
𝑛
(
𝑦
𝑖
−
(
𝑤
𝑥
𝑖
+
𝑏
)
)
2
J(w,b)=
n
1
	​

i=1
∑
n
	​

(y
i
	​

−(wx
i
	​

+b))
2

This loss function has two important properties:

it penalizes large errors more strongly

it is convex, meaning it has a single global minimum

4. Optimization Problem

The goal of training is to solve:

min
⁡
𝑤
,
𝑏
𝐽
(
𝑤
,
𝑏
)
w,b
min
	​

J(w,b)

Rather than solving this analytically (closed-form), we use an iterative optimization algorithm: gradient descent.

5. Gradient Descent

Gradient descent minimizes the loss function by iteratively updating parameters in the direction of steepest descent.

Update rules:
𝑤
←
𝑤
−
𝜂
∂
𝐽
∂
𝑤
w←w−η
∂w
∂J
	​

𝑏
←
𝑏
−
𝜂
∂
𝐽
∂
𝑏
b←b−η
∂b
∂J
	​


Where:

𝜂
η is the learning rate

∇
𝐽
∇J is the gradient of the loss function

6. Derivation of Gradients

Starting from:

𝐽
(
𝑤
,
𝑏
)
=
1
𝑛
∑
(
𝑦
𝑖
−
(
𝑤
𝑥
𝑖
+
𝑏
)
)
2
J(w,b)=
n
1
	​

∑(y
i
	​

−(wx
i
	​

+b))
2
Gradient with respect to 
𝑤
w:
∂
𝐽
∂
𝑤
=
2
𝑛
∑
𝑖
=
1
𝑛
(
𝑤
𝑥
𝑖
+
𝑏
−
𝑦
𝑖
)
𝑥
𝑖
∂w
∂J
	​

=
n
2
	​

i=1
∑
n
	​

(wx
i
	​

+b−y
i
	​

)x
i
	​

Gradient with respect to 
𝑏
b:
∂
𝐽
∂
𝑏
=
2
𝑛
∑
𝑖
=
1
𝑛
(
𝑤
𝑥
𝑖
+
𝑏
−
𝑦
𝑖
)
∂b
∂J
	​

=
n
2
	​

i=1
∑
n
	​

(wx
i
	​

+b−y
i
	​

)

These expressions quantify how changes in parameters affect the loss.

7. Algorithm

The full gradient descent algorithm for linear regression is:

Initialize parameters:

𝑤
=
0
,
𝑏
=
0
w=0,b=0

Repeat for a fixed number of iterations:

Compute predictions:

𝑦
^
𝑖
=
𝑤
𝑥
𝑖
+
𝑏
y
^
	​

i
	​

=wx
i
	​

+b

Compute gradients:

∂
𝐽
∂
𝑤
,
∂
𝐽
∂
𝑏
∂w
∂J
	​

,
∂b
∂J
	​


Update parameters:

𝑤
←
𝑤
−
𝜂
∂
𝐽
∂
𝑤
w←w−η
∂w
∂J
	​

𝑏
←
𝑏
−
𝜂
∂
𝐽
∂
𝑏
b←b−η
∂b
∂J
	​


Return optimized parameters 
𝑤
,
𝑏
w,b



