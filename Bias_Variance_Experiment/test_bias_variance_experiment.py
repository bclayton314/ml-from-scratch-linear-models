import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")  # Prevent GUI backend during tests

from bias_variance_experiment import (
    generate_dataset,
    make_polynomial_features,
    PolynomialRegressionModel,
    fit_polynomial_regression,
    mean_squared_error,
    run_bias_variance_experiment,
    plot_bias_variance_curve,
    plot_sample_fits,
    plot_mean_prediction,
)


def test_generate_dataset_shapes():
    X_train, y_train = generate_dataset(n_samples=25, noise_std=0.1)

    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert X_train.shape == (25,)
    assert y_train.shape == (25,)


def test_generate_dataset_returns_sorted_x():
    X_train, _ = generate_dataset(n_samples=50, noise_std=0.1)
    assert np.all(np.diff(X_train) >= 0)


def test_generate_dataset_is_reproducible_with_fixed_rng():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    X1, y1 = generate_dataset(n_samples=20, noise_std=0.3, rng=rng1)
    X2, y2 = generate_dataset(n_samples=20, noise_std=0.3, rng=rng2)

    assert np.allclose(X1, X2)
    assert np.allclose(y1, y2)


def test_make_polynomial_features_degree_0():
    X = np.array([2.0, 3.0, 4.0])
    X_poly = make_polynomial_features(X, deg=0)

    expected = np.array([
        [1.0],
        [1.0],
        [1.0],
    ])

    assert np.allclose(X_poly, expected)
    assert X_poly.shape == (3, 1)


def test_make_polynomial_features_degree_3():
    X = np.array([2.0, 3.0])
    X_poly = make_polynomial_features(X, deg=3)

    expected = np.array([
        [1.0, 2.0, 4.0, 8.0],
        [1.0, 3.0, 9.0, 27.0],
    ])

    assert np.allclose(X_poly, expected)
    assert X_poly.shape == (2, 4)


def test_polynomial_regression_model_predict():
    # y = 1 + 2x + 3x^2
    weights = np.array([1.0, 2.0, 3.0])
    model = PolynomialRegressionModel(weights=weights, degree=2)

    X = np.array([0.0, 1.0, 2.0])
    X_poly = make_polynomial_features(X, deg=2)
    preds = model.predict(X_poly)

    expected = np.array([1.0, 6.0, 17.0])
    assert np.allclose(preds, expected)


def test_fit_polynomial_regression_learns_exact_quadratic():
    # y = 1 + 2x + 3x^2
    X = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    y = 1.0 + 2.0 * X + 3.0 * (X ** 2)

    model = fit_polynomial_regression(X, y, deg=2)

    assert isinstance(model, PolynomialRegressionModel)
    assert model.degree == 2
    assert np.allclose(model.weights, np.array([1.0, 2.0, 3.0]), atol=1e-10)


def test_fit_polynomial_regression_prediction_matches_training_data():
    X = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    y = 1.0 + 2.0 * X + 3.0 * (X ** 2)

    model = fit_polynomial_regression(X, y, deg=2)
    X_poly = make_polynomial_features(X, deg=2)
    preds = model.predict(X_poly)

    assert np.allclose(preds, y, atol=1e-10)


def test_mean_squared_error_zero_for_identical_arrays():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    mse = mean_squared_error(y_true, y_pred)
    assert np.isclose(mse, 0.0)


def test_mean_squared_error_correct_value():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 4.0])

    # errors: [1, 0, 1], squared mean = 2/3
    mse = mean_squared_error(y_true, y_pred)
    assert np.isclose(mse, 2.0 / 3.0)


def test_run_bias_variance_experiment_output_structure():
    degrees = [1, 3]

    x_test, y_true_test, results, sample_curves = run_bias_variance_experiment(
        degrees=degrees,
        n_trials=5,
        n_train_samples=10,
        noise_std=0.1,
        x_range=(-2, 2),
        random_seed=123,
    )

    assert x_test.shape == (200,)
    assert y_true_test.shape == (200,)
    assert isinstance(results, dict)
    assert isinstance(sample_curves, dict)

    for degree in degrees:
        assert degree in results
        assert degree in sample_curves

        assert "bias2" in results[degree]
        assert "variance" in results[degree]
        assert "avg_test_error" in results[degree]
        assert "mean_prediction" in results[degree]
        assert "all_predictions" in results[degree]

        assert results[degree]["mean_prediction"].shape == (200,)
        assert results[degree]["all_predictions"].shape == (5, 200)

        assert "x_train" in sample_curves[degree]
        assert "y_train" in sample_curves[degree]
        assert "sample_predictions" in sample_curves[degree]

        assert sample_curves[degree]["x_train"].shape == (10,)
        assert sample_curves[degree]["y_train"].shape == (10,)
        assert sample_curves[degree]["sample_predictions"].shape == (5, 200)


def test_run_bias_variance_experiment_reproducible_with_seed():
    degrees = [1, 3]

    out1 = run_bias_variance_experiment(
        degrees=degrees,
        n_trials=5,
        n_train_samples=10,
        noise_std=0.1,
        x_range=(-2, 2),
        random_seed=999,
    )

    out2 = run_bias_variance_experiment(
        degrees=degrees,
        n_trials=5,
        n_train_samples=10,
        noise_std=0.1,
        x_range=(-2, 2),
        random_seed=999,
    )

    x_test_1, y_true_1, results_1, sample_curves_1 = out1
    x_test_2, y_true_2, results_2, sample_curves_2 = out2

    assert np.allclose(x_test_1, x_test_2)
    assert np.allclose(y_true_1, y_true_2)

    for degree in degrees:
        assert np.allclose(results_1[degree]["mean_prediction"], results_2[degree]["mean_prediction"])
        assert np.allclose(results_1[degree]["all_predictions"], results_2[degree]["all_predictions"])
        assert np.isclose(results_1[degree]["bias2"], results_2[degree]["bias2"])
        assert np.isclose(results_1[degree]["variance"], results_2[degree]["variance"])
        assert np.isclose(results_1[degree]["avg_test_error"], results_2[degree]["avg_test_error"])

        assert np.allclose(sample_curves_1[degree]["x_train"], sample_curves_2[degree]["x_train"])
        assert np.allclose(sample_curves_1[degree]["y_train"], sample_curves_2[degree]["y_train"])
        assert np.allclose(sample_curves_1[degree]["sample_predictions"], sample_curves_2[degree]["sample_predictions"])


def test_sample_predictions_capped_at_10():
    degrees = [2]

    _, _, _, sample_curves = run_bias_variance_experiment(
        degrees=degrees,
        n_trials=25,
        n_train_samples=8,
        noise_std=0.1,
        random_seed=123,
    )

    assert sample_curves[2]["sample_predictions"].shape == (10, 200)


def test_plot_bias_variance_curve_runs(monkeypatch):
    called = {"show": 0}

    def fake_show():
        called["show"] += 1

    monkeypatch.setattr("matplotlib.pyplot.show", fake_show)

    results = {
        1: {"bias2": 0.5, "variance": 0.1, "avg_test_error": 0.6},
        3: {"bias2": 0.2, "variance": 0.2, "avg_test_error": 0.4},
    }

    plot_bias_variance_curve([1, 3], results)

    assert called["show"] == 1


def test_plot_sample_fits_runs(monkeypatch):
    called = {"show": 0}

    def fake_show():
        called["show"] += 1

    monkeypatch.setattr("matplotlib.pyplot.show", fake_show)

    x_test = np.linspace(-3, 3, 200)
    y_true_test = np.sin(x_test)

    sample_curves = {
        1: {
            "x_train": np.array([-1.0, 0.0, 1.0]),
            "y_train": np.array([-0.8, 0.0, 0.8]),
            "sample_predictions": np.array([
                np.sin(x_test),
                0.9 * np.sin(x_test),
            ]),
        }
    }

    plot_sample_fits(x_test, y_true_test, sample_curves, degrees_to_plot=[1])

    assert called["show"] == 1


def test_plot_mean_prediction_runs(monkeypatch):
    called = {"show": 0}

    def fake_show():
        called["show"] += 1

    monkeypatch.setattr("matplotlib.pyplot.show", fake_show)

    x_test = np.linspace(-3, 3, 200)
    y_true_test = np.sin(x_test)

    results = {
        5: {
            "mean_prediction": 0.95 * np.sin(x_test)
        }
    }

    plot_mean_prediction(x_test, y_true_test, results, degree=5)

    assert called["show"] == 1

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
    