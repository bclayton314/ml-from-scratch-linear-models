import numpy as np

from LinearRegression_ClosedF import LinearRegression
from LinearRegression_GradD import LinearRegressionGD


def main():
    # Simple linear relationship: y = 1 + 2x
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ])

    y = np.array([3.0, 5.0, 7.0, 9.0, 11.0])

    # Closed-form model
    closed_form_model = LinearRegression(
        fit_intercept=True,
        method="pinv",
    )
    closed_form_model.fit(X, y)

    # Gradient descent model
    gd_model = LinearRegressionGD(
        fit_intercept=True,
        learn_rate=0.05,
        n_iters=5000,
    )
    gd_model.fit(X, y)

    # Predictions
    y_pred_closed = closed_form_model.predict(X)
    y_pred_gd = gd_model.predict(X)

    print("=== Closed-Form Solution ===")
    print("Intercept:", closed_form_model.intercept_)
    print("Coefficient(s):", closed_form_model.coef_)
    print("Predictions:", y_pred_closed)
    print("MSE:", closed_form_model.mse(X, y))
    print()

    print("=== Gradient Descent Solution ===")
    print("Intercept:", gd_model.intercept_)
    print("Coefficient(s):", gd_model.coef_)
    print("Predictions:", y_pred_gd)
    print("MSE:", gd_model.mse(X, y))
    print()

    print("=== Differences ===")
    print("Intercept difference:",
          abs(closed_form_model.intercept_ - gd_model.intercept_))
    print("Coefficient difference:",
          np.abs(closed_form_model.coef_ - gd_model.coef_))
    print("Prediction difference:",
          np.abs(y_pred_closed - y_pred_gd))
    print()

    print("All predictions close?",
          np.allclose(y_pred_closed, y_pred_gd, atol=1e-4))
    print("All coefficients close?",
          np.allclose(closed_form_model.coef_, gd_model.coef_, atol=1e-4))
    print("Intercept close?",
          np.isclose(closed_form_model.intercept_, gd_model.intercept_, atol=1e-4))


if __name__ == "__main__":
    main()