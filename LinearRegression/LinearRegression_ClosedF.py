import numpy as np
from typing import Literal


class LinearRegression:
    """
    Linear Regression from scratch using NumPy.

    Supports:
    - pseudoinverse solution (recommended)
    - normal equation solution

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to fit an intercept term.

    method : {"pinv", "normal"}, default="pinv"
        Solver to use:
        - "pinv": Moore-Penrose pseudoinverse (more numerically stable)
        - "normal": Normal equation using inverse of X^T X

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Learned feature weights.

    intercept_ : float
        Learned intercept.

    fitted_ : bool
        Whether the model has been fit.
    """
    def __init__(self, fit_intercept: bool=True, method: Literal["pinv", "normal"] = "pinv") -> None:
        if method not in {"pinv", "normal"}:
            raise ValueError("method must be either 'pinv' or 'normal'")
        self.fit_intercept = fit_intercept
        self.method = method
        self.coef_ = None
        self.intercept_ = 0.0
        self.fitted_ = False

    def _add_intercept_column(self, X):
        ones = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack((ones, X))

    def solve_weights(self, X, y):
        if self.method == "pinv":
            return np.linalg.pinv(X) @ y

        if self.method == "normal":
            return np.linalg.inv(X.T @ X) @ (X.T) @ y

    def fit(self, X, y):
        if self.fit_intercept:
            X_intercept = self._add_intercept_column(X)
        else:
            X_intercept = X
        
        w = self.solve_weights(X_intercept, y)

        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w

        self.fitted_ = True
        return self

    def predict(self, X):
        if not self.fitted_:
            raise ValueError("Model has not been fit yet")
        
        return X @ self.coef_ + self.intercept_
    
    # be sure to post a copy of R^2 formula and explanation
    def r_squared(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return float(1.0 - ss_res / ss_tot)

    def mse(self, X, y):
        y_pred = self.predict(X)
        return float(np.mean((y - y_pred) ** 2))




