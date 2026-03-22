import numpy as np
import matplotlib.pyplot as plt


def generate_dataset(n_samples=30, noise_std=0.3, x_range=(-3, 3), rng=None):
    if rng is None:
        rng = np.random.default_rng()

    X_train = rng.uniform(x_range[0], x_range[1], size=n_samples)
    X_train = np.sort(X_train)

    y_true = np.sin(X_train)
    y_train = y_true + rng.normal(0.0, noise_std, size=n_samples)

    return X_train, y_train


def make_polynomial_features(X, deg):
    X = np.asarray(X)
    return np.column_stack([X ** p for p in range(deg + 1)])


class PolynomialRegressionModel:
    def __init__(self, weights, degree):
        self.weights = weights
        self.degree = degree

    def predict(self, X_poly):
        return X_poly @ self.weights


def fit_polynomial_regression(X, y, deg):
    X_poly = make_polynomial_features(X, deg)
    weights = np.linalg.pinv(X_poly) @ y
    return PolynomialRegressionModel(weights, deg)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def run_bias_variance_experiment(
    degrees,
    n_trials=200,
    n_train_samples=30,
    noise_std=0.3,
    x_range=(-3, 3),
    random_seed=42,
):
    rng = np.random.default_rng(random_seed)

    x_test = np.linspace(x_range[0], x_range[1], 200)
    y_true_test = np.sin(x_test)

    results = {}
    sample_curves = {}

    for degree in degrees:
        all_predictions = []
        all_test_mse = []

        X_test_poly = make_polynomial_features(x_test, degree)

        sample_predictions = []
        sample_X_train = None
        sample_y_train = None

        for trial in range(n_trials):
            X_train, y_train = generate_dataset(
                n_samples=n_train_samples,
                noise_std=noise_std,
                x_range=x_range,
                rng=rng,
            )

            model = fit_polynomial_regression(X_train, y_train, degree)
            y_pred_test = model.predict(X_test_poly)

            all_predictions.append(y_pred_test)

            mse = mean_squared_error(y_true_test, y_pred_test)
            all_test_mse.append(mse)

            if trial < 10:
                sample_predictions.append(y_pred_test)

            if trial == 0:
                sample_X_train = X_train
                sample_y_train = y_train

        all_predictions = np.array(all_predictions)
        mean_prediction = all_predictions.mean(axis=0)

        bias2 = np.mean((mean_prediction - y_true_test) ** 2)
        variance = np.mean(np.var(all_predictions, axis=0))
        avg_test_error = np.mean(all_test_mse)

        results[degree] = {
            "bias2": bias2,
            "variance": variance,
            "avg_test_error": avg_test_error,
            "mean_prediction": mean_prediction,
            "all_predictions": all_predictions,
        }

        sample_curves[degree] = {
            "x_train": sample_X_train,
            "y_train": sample_y_train,
            "sample_predictions": np.array(sample_predictions),
        }

        print(
            f"Degree={degree:>2} | "
            f"Bias^2={bias2:.6f} | "
            f"Variance={variance:.6f} | "
            f"Avg Error={avg_test_error:.6f}"
        )

    return x_test, y_true_test, results, sample_curves


def plot_bias_variance_curve(degrees, results):
    bias2_vals = [results[d]["bias2"] for d in degrees]
    variance_vals = [results[d]["variance"] for d in degrees]
    error_vals = [results[d]["avg_test_error"] for d in degrees]

    plt.figure(figsize=(10, 6))
    plt.plot(degrees, bias2_vals, marker="o", label="Bias^2")
    plt.plot(degrees, variance_vals, marker="o", label="Variance")
    plt.plot(degrees, error_vals, marker="o", label="Average Error")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Error")
    plt.title("Bias-Variance Tradeoff")
    plt.xticks(degrees)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_sample_fits(x_test, y_true_test, sample_curves, degrees_to_plot):
    for degree in degrees_to_plot:
        curves = sample_curves[degree]

        plt.figure(figsize=(10, 6))
        plt.plot(x_test, y_true_test, linewidth=2, label="True Function: sin(x)")
        plt.scatter(
            curves["x_train"],
            curves["y_train"],
            s=30,
            alpha=0.7,
            label="Training Data",
        )

        for pred in curves["sample_predictions"]:
            plt.plot(x_test, pred, alpha=0.35)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Sample Fits for Polynomial Degree {degree}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_mean_prediction(x_test, y_true_test, results, degree):
    mean_prediction = results[degree]["mean_prediction"]

    plt.figure(figsize=(10, 6))
    plt.plot(x_test, y_true_test, linewidth=2, label="True Function: sin(x)")
    plt.plot(x_test, mean_prediction, linestyle="--", linewidth=2, label="Mean Prediction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Mean Prediction vs True Function (Degree {degree})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    degrees = [1, 3, 5, 9, 15]

    x_test, y_true_test, results, sample_curves = run_bias_variance_experiment(
        degrees=degrees,
        n_trials=200,
        n_train_samples=30,
        noise_std=0.3,
        x_range=(-3, 3),
        random_seed=42,
    )

    plot_bias_variance_curve(degrees, results)

    plot_sample_fits(x_test, y_true_test, sample_curves, degrees_to_plot=[1, 15])

    plot_mean_prediction(x_test, y_true_test, results, degree=5)
