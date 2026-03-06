import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data.csv')

data = pd.read_csv(data_path)

plt.scatter(data['x'], data['y'])
plt.show()

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



def loss_function(m, b, points):
    pass


def gradient_descent(m_current, b_current, points, L):
    pass

