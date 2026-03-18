import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data.csv')

data = pd.read_csv(data_path)

"""
linear_regression_plot.py

"""

class LinearRegressionGD:
    def __init__(self, fit_intercept: bool = True, learn_rate: int = 0.01, n_iters: int = 1000) -> None:
        self.fit_intercept = fit_intercept
        self.learn_rate = learn_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _add_intercept_column(self, X):
        ones = np.ones((X.shape[0], 1), dtype=np.float64)
        return np.hstack((ones, X))
    
    def gradient_descent(self, X, y):
        # X = np.asarray(X, dtype=np.float64)
        # y = np.asarray(y, dtype=np.float64)

        # if y.ndim > 1:
        #     y = y.ravel()

        if self.fit_intercept:
            X_train = self._add_intercept_column(X)
        else:
            X_train = X

        n_samples, n_features = X_train.shape

        self.weights = np.zeros(n_features, dtype=np.float64)

        for _ in range(self.n_iters):
            y_pred = X_train @ self.weights
            error = y_pred - y

            grad = (2.0 / n_samples) * (X_train.T @ error)
            self.weights -= self.learn_rate * grad

        if self.fit_intercept:
            self.intercept_ = float(self.weights[0])
            self.coef_ = self.weights[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = self.weights.copy()

        return self

    def fit(self, X, y):
        return self.gradient_descent(X, y)

    def predict(self, X):
        # X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_

    def mse(self, X, y):
        y_pred = self.predict(X)
        return float(np.mean((y - y_pred) ** 2))

    def r_squared(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return float(1.0 - ss_res / ss_tot)



# def gradient_descent(m_current, b_current, points, l_rate):
#     m_grad = 0
#     b_grad = 0

#     n = len(points)

#     for i in range(n):
#         x = points.iloc[i].x
#         y = points.iloc[i].y

#         m_grad += -2/n * x * (y - (m_current * x + b_current))
#         b_grad += -2/n * (y - (m_current * x + b_current))

#     m = m_current - m_grad * l_rate
#     b = b_current - b_grad * l_rate

#     return m, b

# m = 0
# b = 0
# l_rate = 0.0001
# epochs = 200

# for i in range(epochs):
#     if i % 50 == 0:
#         print(f"Epoch: {i}")
#     m, b = gradient_descent(m, b, data, l_rate)

# print(m, b)

# plt.scatter(data.x, data.y, color="black")
# plt.plot(list(range(0, 100)), [m * x + b for x in range(0, 100)], color="red")
# plt.show()


# # Hook this up to class
# # -unit_tests
# # -do math beforehand tomorrow