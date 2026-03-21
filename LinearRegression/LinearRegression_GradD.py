import numpy as np


class LinearRegressionGD:
    """
    Linear Regression from scratch using Gradient Descent.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to fit an intercept term.

    learn_rate : float, default=0.01
        Step size for gradient descent.

    n_iters : int, default=1000
        Number of passes through the training data.

    """
    def __init__(self, fit_intercept: bool = True, learn_rate: float = 0.01, n_iters: int = 1000) -> None:
        self.fit_intercept = fit_intercept
        self.learn_rate = learn_rate
        self.n_iters = n_iters
        self.weights = None

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
        if self.weights is None:
            raise ValueError("Model has not been fit yet")
        return X @ self.coef_ + self.intercept_

    def mse(self, X, y):
        y_pred = self.predict(X)
        return float(np.mean((y - y_pred) ** 2))

    def r_squared(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return float(1.0 - ss_res / ss_tot)

