import numpy as np
import pytest

from LinearRegression_ClosedF import LinearRegression


def test_invalid_method_raises_value_error():
    with pytest.raises(ValueError, match="method must be either 'pinv' or 'normal'"):
        LinearRegression(method="bad_method")


def test_default_init_values():
    model = LinearRegression()

    assert model.fit_intercept is True
    assert model.method == "pinv"
    assert model.coef_ is None
    assert model.intercept_ == 0.0
    assert model.fitted_ is False


def test_add_intercept_column():
    model = LinearRegression()

    X = np.array([
        [2.0, 3.0],
        [4.0, 5.0]
    ])

    X_new = model._add_intercept_column(X)

    expected = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 4.0, 5.0]
    ])

    assert np.allclose(X_new, expected)


def test_predict_before_fit_raises_error():
    model = LinearRegression()
    X = np.array([[1.0], [2.0]])

    with pytest.raises(ValueError, match="Model has not been fit yet"):
        model.predict(X)


def test_fit_sets_fitted_flag():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([3.0, 5.0, 7.0])

    model = LinearRegression()
    model.fit(X, y)

    assert model.fitted_ is True


def test_fit_with_intercept_pinv_learns_simple_line():
    # y = 2x + 1
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])

    model = LinearRegression(fit_intercept=True, method="pinv")
    model.fit(X, y)

    assert np.allclose(model.coef_, np.array([2.0]), atol=1e-10)
    assert np.isclose(model.intercept_, 1.0, atol=1e-10)


def test_fit_with_intercept_normal_learns_simple_line():
    # y = 2x + 1
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])

    model = LinearRegression(fit_intercept=True, method="normal")
    model.fit(X, y)

    assert np.allclose(model.coef_, np.array([2.0]), atol=1e-10)
    assert np.isclose(model.intercept_, 1.0, atol=1e-10)


def test_fit_without_intercept():
    # y = 3x
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 6.0, 9.0, 12.0])

    model = LinearRegression(fit_intercept=False, method="pinv")
    model.fit(X, y)

    assert np.allclose(model.coef_, np.array([3.0]), atol=1e-10)
    assert np.isclose(model.intercept_, 0.0, atol=1e-10)


def test_predict_returns_expected_values():
    # y = 2x + 1
    X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_train = np.array([3.0, 5.0, 7.0, 9.0])

    X_test = np.array([[5.0], [6.0]])

    model = LinearRegression(fit_intercept=True, method="pinv")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    expected = np.array([11.0, 13.0])
    assert np.allclose(preds, expected, atol=1e-10)


def test_mse_is_zero_for_perfect_fit():
    # y = 2x + 1
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])

    model = LinearRegression(method="pinv")
    model.fit(X, y)

    mse = model.mse(X, y)
    assert np.isclose(mse, 0.0, atol=1e-12)


def test_r_squared_is_one_for_perfect_fit():
    # y = 2x + 1
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])

    model = LinearRegression(method="pinv")
    model.fit(X, y)

    r2 = model.r_squared(X, y)
    assert np.isclose(r2, 1.0, atol=1e-12)


def test_pinv_and_normal_give_same_predictions_on_clean_data():
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0]
    ])
    y = np.array([3.0, 5.0, 7.0, 9.0, 11.0])  # y = 2x + 1

    model_pinv = LinearRegression(method="pinv")
    model_normal = LinearRegression(method="normal")

    model_pinv.fit(X, y)
    model_normal.fit(X, y)

    preds_pinv = model_pinv.predict(X)
    preds_normal = model_normal.predict(X)

    assert np.allclose(preds_pinv, preds_normal, atol=1e-10)
    assert np.allclose(model_pinv.coef_, model_normal.coef_, atol=1e-10)
    assert np.isclose(model_pinv.intercept_, model_normal.intercept_, atol=1e-10)


def test_multifeature_fit():
    # y = 1 + 2*x1 + 3*x2
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 0.0],
        [0.0, 3.0],
        [4.0, 2.0]
    ])
    y = 1.0 + 2.0 * X[:, 0] + 3.0 * X[:, 1]

    model = LinearRegression(fit_intercept=True, method="pinv")
    model.fit(X, y)

    assert np.allclose(model.coef_, np.array([2.0, 3.0]), atol=1e-10)
    assert np.isclose(model.intercept_, 1.0, atol=1e-10)


def test_solve_weights_returns_correct_shape_with_intercept():
    X = np.array([
        [1.0],
        [2.0],
        [3.0]
    ])
    y = np.array([3.0, 5.0, 7.0])

    model = LinearRegression(method="pinv")
    X_intercept = model._add_intercept_column(X)
    w = model.solve_weights(X_intercept, y)

    assert w.shape == (2,)


def test_normal_method_raises_on_singular_matrix():
    X = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0]
    ])
    y = np.array([1.0, 2.0, 3.0])

    model = LinearRegression(method="normal")

    with pytest.raises(np.linalg.LinAlgError):
        model.fit(X, y)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
