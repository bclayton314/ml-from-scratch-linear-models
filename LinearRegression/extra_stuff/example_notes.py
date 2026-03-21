


# LINREG (CLOSED FORM) EXAMPLE FROM CHATGPT
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray


ArrayLike = NDArray[np.float64]


@dataclass
class FitResult:
    """Optional container for fit diagnostics."""
    n_samples: int
    n_features: int
    rank: int
    method: str


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

    fit_result_ : FitResult
        Diagnostic metadata from fitting.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        method: Literal["pinv", "normal"] = "pinv",
    ) -> None:
        if method not in {"pinv", "normal"}:
            raise ValueError("method must be either 'pinv' or 'normal'")

        self.fit_intercept = fit_intercept
        self.method = method

        self.coef_: Optional[ArrayLike] = None
        self.intercept_: float = 0.0
        self.fitted_: bool = False
        self.fit_result_: Optional[FitResult] = None

    def _validate_inputs(self, X: ArrayLike, y: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """Validate and coerce X and y into proper NumPy float arrays."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of rows. Got X.shape[0]={X.shape[0]} "
                f"and y.shape[0]={y.shape[0]}"
            )

        if X.shape[0] == 0:
            raise ValueError("X and y must contain at least one sample")

        return X, y

    def _validate_predict_input(self, X: ArrayLike) -> ArrayLike:
        """Validate X for prediction."""
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        if self.coef_ is None:
            raise ValueError("Model has not been fit yet")

        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, but model expects {self.coef_.shape[0]}"
            )

        return X

    def _add_intercept_column(self, X: ArrayLike) -> ArrayLike:
        """Add a leading column of ones for intercept fitting."""
        ones = np.ones((X.shape[0], 1), dtype=np.float64)
        return np.hstack((ones, X))

    def _solve_weights(self, X_design: ArrayLike, y: ArrayLike) -> ArrayLike:
        """Solve for weights using the selected method."""
        if self.method == "pinv":
            return np.linalg.pinv(X_design) @ y

        # method == "normal"
        xtx = X_design.T @ X_design
        xty = X_design.T @ y

        try:
            return np.linalg.inv(xtx) @ xty
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                "Normal equation failed because X^T X is singular or not invertible. "
                "Try method='pinv' instead."
            ) from exc

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LinearRegression":
        """
        Fit the linear regression model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.

        y : np.ndarray of shape (n_samples,)
            Target vector.

        Returns
        -------
        self : LinearRegression
            Fitted model.
        """
        X, y = self._validate_inputs(X, y)

        X_design = self._add_intercept_column(X) if self.fit_intercept else X
        w = self._solve_weights(X_design, y)

        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = w.copy()

        self.fitted_ = True
        self.fit_result_ = FitResult(
            n_samples=X.shape[0],
            n_features=X.shape[1],
            rank=int(np.linalg.matrix_rank(X_design)),
            method=self.method,
        )

        return self

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict target values for input X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted target values.
        """
        if not self.fitted_:
            raise ValueError("Model has not been fit yet")

        X = self._validate_predict_input(X)
        return X @ self.coef_ + self.intercept_

    def score_r2(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Compute R^2 score.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.

        y : np.ndarray of shape (n_samples,)
            True targets.

        Returns
        -------
        float
            R^2 score.
        """
        X = self._validate_predict_input(X)
        y = np.asarray(y, dtype=np.float64)

        if y.ndim != 1:
            raise ValueError("y must be 1D")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if np.isclose(ss_tot, 0.0):
            return 1.0 if np.isclose(ss_res, 0.0) else 0.0

        return float(1.0 - ss_res / ss_tot)

    def mse(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Compute mean squared error.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.

        y : np.ndarray of shape (n_samples,)
            True targets.

        Returns
        -------
        float
            Mean squared error.
        """
        X = self._validate_predict_input(X)
        y = np.asarray(y, dtype=np.float64)

        if y.ndim != 1:
            raise ValueError("y must be 1D")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        y_pred = self.predict(X)
        return float(np.mean((y - y_pred) ** 2))

    def residuals(self, X: ArrayLike, y: ArrayLike) -> ArrayLike:
        """
        Compute residuals: y - y_pred.
        """
        X = self._validate_predict_input(X)
        y = np.asarray(y, dtype=np.float64)

        if y.ndim != 1:
            raise ValueError("y must be 1D")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        return y - self.predict(X)

    def get_params(self) -> dict[str, object]:
        """Return model configuration."""
        return {
            "fit_intercept": self.fit_intercept,
            "method": self.method,
        }

    def __repr__(self) -> str:
        status = "fitted" if self.fitted_ else "not fitted"
        return (
            f"LinearRegression(fit_intercept={self.fit_intercept}, "
            f"method='{self.method}', status='{status}')"
        )



# LINREG (GRAD DESCENT) EXAMPLE FROM CHATGPT
from __future__ import annotations
import numpy as np

class LinearRegressionGD:
    """
    Linear Regression from scratch using Gradient Descent.

    Supports:
    - batch gradient descent
    - mini-batch gradient descent

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to fit an intercept term.

    learning_rate : float, default=0.01
        Step size for gradient descent.

    epochs : int, default=1000
        Number of passes through the training data.

    batch_size : int | None, default=None
        If None, use full batch gradient descent.
        If an integer, use mini-batch gradient descent.

    tolerance : float, default=1e-8
        Early stopping threshold based on change in loss.

    random_state : int | None, default=None
        Seed for reproducible mini-batch shuffling.

    verbose : bool, default=False
        Whether to print training progress.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        batch_size: int | None = None,
        tolerance: float = 1e-8,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if epochs <= 0:
            raise ValueError("epochs must be > 0")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be > 0 if provided")
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0")

        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.random_state = random_state
        self.verbose = verbose

        self.coef_ = None
        self.intercept_ = 0.0
        self.loss_history_ = []
        self.n_features_in_ = None

    def _validate_inputs(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got {X.shape[0]} and {y.shape[0]}"
            )
        if X.shape[0] == 0:
            raise ValueError("X and y must contain at least one sample")

        return X, y

    def _validate_predict_input(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if self.coef_ is None:
            raise ValueError("Model has not been fit yet")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model expects {self.n_features_in_}"
            )

        return X

    def _add_intercept_column(self, X):
        ones = np.ones((X.shape[0], 1), dtype=np.float64)
        return np.hstack((ones, X))

    def _compute_loss(self, X_design, y, weights):
        y_pred = X_design @ weights
        return np.mean((y - y_pred) ** 2)

    def _compute_gradient(self, X_batch, y_batch, weights):
        n = X_batch.shape[0]
        return (2.0 / n) * X_batch.T @ (X_batch @ weights - y_batch)

    def fit(self, X, y):
        """
        Fit the model using gradient descent.
        """
        X, y = self._validate_inputs(X, y)
        self.n_features_in_ = X.shape[1]

        X_design = self._add_intercept_column(X) if self.fit_intercept else X
        n_samples, n_features = X_design.shape

        rng = np.random.default_rng(self.random_state)
        weights = np.zeros(n_features, dtype=np.float64)

        self.loss_history_ = []
        prev_loss = None

        for epoch in range(self.epochs):
            if self.batch_size is None:
                gradient = self._compute_gradient(X_design, y, weights)
                weights -= self.learning_rate * gradient
            else:
                indices = rng.permutation(n_samples)
                X_shuffled = X_design[indices]
                y_shuffled = y[indices]

                for start in range(0, n_samples, self.batch_size):
                    end = start + self.batch_size
                    X_batch = X_shuffled[start:end]
                    y_batch = y_shuffled[start:end]

                    gradient = self._compute_gradient(X_batch, y_batch, weights)
                    weights -= self.learning_rate * gradient

            loss = self._compute_loss(X_design, y, weights)
            self.loss_history_.append(loss)

            if self.verbose and (epoch % 100 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch:4d} | Loss: {loss:.8f}")

            if prev_loss is not None and abs(prev_loss - loss) < self.tolerance:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

            prev_loss = loss

        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = weights[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = weights.copy()

        return self

    def predict(self, X):
        """
        Predict target values.
        """
        X = self._validate_predict_input(X)
        return X @ self.coef_ + self.intercept_

    def mse(self, X, y):
        """
        Compute mean squared error.
        """
        X = self._validate_predict_input(X)
        y = np.asarray(y, dtype=np.float64)

        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        y_pred = self.predict(X)
        return float(np.mean((y - y_pred) ** 2))

    def residuals(self, X, y):
        """
        Compute residuals: y - y_pred.
        """
        X = self._validate_predict_input(X)
        y = np.asarray(y, dtype=np.float64)

        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        return y - self.predict(X)

    def score_r2(self, X, y):
        """
        Compute R^2 score.
        """
        X = self._validate_predict_input(X)
        y = np.asarray(y, dtype=np.float64)

        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if np.isclose(ss_tot, 0.0):
            return 1.0 if np.isclose(ss_res, 0.0) else 0.0

        return float(1.0 - ss_res / ss_tot)

    def get_params(self):
        return {
            "fit_intercept": self.fit_intercept,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "tolerance": self.tolerance,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def __repr__(self):
        status = "fitted" if self.coef_ is not None else "not fitted"
        return (
            f"LinearRegressionGD("
            f"fit_intercept={self.fit_intercept}, "
            f"learning_rate={self.learning_rate}, "
            f"epochs={self.epochs}, "
            f"batch_size={self.batch_size}, "
            f"status='{status}')"
        )


