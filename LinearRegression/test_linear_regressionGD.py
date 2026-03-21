import numpy as np
import pytest

from LinearRegression_GradD import LinearRegressionGD


def test_default_init_values():
    model = LinearRegressionGD()

    assert model.fit_intercept is True
    assert model.learn_rate == 0.01
    assert model.n_iters == 1000
    assert model.weights is None


def test_add_intercept_column():
    model = LinearRegressionGD()

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


def test_fit_returns_self():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])

    model = LinearRegressionGD(learn_rate=0.01, n_iters=5000)
    returned = model.fit(X, y)

    assert returned is model


def test_gradient_descent_returns_self():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])

    model = LinearRegressionGD(learn_rate=0.01, n_iters=5000)
    returned = model.gradient_descent(X, y)

    assert returned is model


def test_weights_change_after_training():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])

    model = LinearRegressionGD(learn_rate=0.01, n_iters=1000)
    model.fit(X, y)

    assert model.weights is not None
    assert not np.allclose(model.weights, 0.0)


def test_fit_with_intercept_learns_simple_line():
    # y = 2x + 1
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])

    model = LinearRegressionGD(fit_intercept=True, learn_rate=0.01, n_iters=10000)
    model.fit(X, y)

    assert np.allclose(model.coef_, np.array([2.0]), atol=1e-2)
    assert np.isclose(model.intercept_, 1.0, atol=1e-2)


def test_fit_without_intercept_learns_simple_line():
    # y = 3x
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 6.0, 9.0, 12.0])

    model = LinearRegressionGD(fit_intercept=False, learn_rate=0.01, n_iters=10000)
    model.fit(X, y)

    assert np.allclose(model.coef_, np.array([3.0]), atol=1e-2)
    assert np.isclose(model.intercept_, 0.0, atol=1e-10)


def test_predict_returns_expected_values():
    # y = 2x + 1
    X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_train = np.array([3.0, 5.0, 7.0, 9.0])

    X_test = np.array([[5.0], [6.0]])

    model = LinearRegressionGD(fit_intercept=True, learn_rate=0.01, n_iters=10000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    expected = np.array([11.0, 13.0])

    assert np.allclose(preds, expected, atol=1e-1)


def test_mse_is_near_zero_for_perfect_linear_data():
    # y = 2x + 1
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])

    model = LinearRegressionGD(fit_intercept=True, learn_rate=0.01, n_iters=10000)
    model.fit(X, y)

    mse = model.mse(X, y)
    assert mse < 1e-3


def test_r_squared_is_near_one_for_perfect_linear_data():
    # y = 2x + 1
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])

    model = LinearRegressionGD(fit_intercept=True, learn_rate=0.01, n_iters=10000)
    model.fit(X, y)

    r2 = model.r_squared(X, y)
    assert np.isclose(r2, 1.0, atol=1e-3)


def test_multifeature_fit():
    # y = 1 + 2*x1 + 3*x2
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 0.0],
        [0.0, 3.0],
        [4.0, 2.0],
        [2.0, 4.0]
    ])
    y = 1.0 + 2.0 * X[:, 0] + 3.0 * X[:, 1]

    model = LinearRegressionGD(fit_intercept=True, learn_rate=0.01, n_iters=20000)
    model.fit(X, y)

    assert np.allclose(model.coef_, np.array([2.0, 3.0]), atol=1e-1)
    assert np.isclose(model.intercept_, 1.0, atol=1e-1)


def test_predictions_have_correct_shape():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])

    model = LinearRegressionGD(fit_intercept=True, learn_rate=0.01, n_iters=5000)
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == y.shape


def test_training_improves_mse():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])

    weak_model = LinearRegressionGD(fit_intercept=True, learn_rate=0.01, n_iters=1)
    strong_model = LinearRegressionGD(fit_intercept=True, learn_rate=0.01, n_iters=10000)

    weak_model.fit(X, y)
    strong_model.fit(X, y)

    weak_mse = weak_model.mse(X, y)
    strong_mse = strong_model.mse(X, y)

    assert strong_mse < weak_mse

def test_predict_before_fit_raises_error():
    model = LinearRegressionGD()
    X = np.array([[1.0], [2.0]])

    with pytest.raises(ValueError, match="Model has not been fit yet"):
        model.predict(X)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
