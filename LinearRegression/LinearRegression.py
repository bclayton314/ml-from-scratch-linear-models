import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data.csv')

data = pd.read_csv(data_path)

# plt.scatter(data['x'], data['y'])
# plt.show()

"""
linear_regression_plot.py

"""

class LinearRegression:
    def __init__(self, learn_rate: int = 0.01, n_iters: int = 1000) -> None:
        self.learn_rate = learn_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.random.rand(num_features)
        self.bias = 0

        for i in range(self.n_iters):

            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / num_samples) * np.dot(X.T, y_pred - y)
            db = (1 / num_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


#loss = [yi - (m*xi + b)]**2
# def loss_function(m, b, points):
#     total_error = 0
#     for i in range(len(points)):
#         x = points.iloc[i].x
#         y = points.iloc[i].y

#         total_error += (y - (m * x + b))**2


def gradient_descent(m_current, b_current, points, l_rate):
    m_grad = 0
    b_grad = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].x
        y = points.iloc[i].y

        m_grad += -2/n * x * (y - (m_current * x + b_current))
        b_grad += -2/n * (y - (m_current * x + b_current))

    m = m_current - m_grad * l_rate
    b = b_current - b_grad * l_rate

    return m, b

m = 0
b = 0
l_rate = 0.0001
epochs = 200

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, data, l_rate)

print(m, b)

plt.scatter(data.x, data.y, color="black")
plt.plot(list(range(0, 100)), [m * x + b for x in range(0, 100)], color="red")
plt.show()


# Hook this up to class
# -unit_tests
# -do math beforehand tomorrow