import numpy as np



def generate_dataset():
    x_test = np.linspace(-3, 3, 200)
    y_true = np.sin(x_test)
    pass

def fit_polynomial_regression(X, y, deg):
    pass

def mean_squared_error(y_test_noisy, y_pred_test):
    pass

for degree in degrees:
    all_predictions = []
    all_test_mse = []

    for trial in range(n_trials):
        X_train, y_train = generate_dataset()
        model = fit_polynomial_regression(X_train, y_train, degree)

        y_pred_test = model.predict(X_test_poly)
        all_predictions.append(y_pred_test)

        mse = mean_squared_error(y_test_noisy_or_true, y_pred_test)
        all_test_mse.append(mse)

    all_predictions = np.array(all_predictions)

    mean_prediction = all_predictions.mean(axis=0)

    bias2 = np.mean((mean_prediction - y_true_test) ** 2)
    variance = np.mean(np.var(all_predictions, axis=0))
    avg_test_error = np.mean(all_test_mse)


###############################################################
###############################################################
import numpy as np


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


degrees = [1, 3, 5, 9, 15]
n_trials = 200
n_train_samples = 30
noise_std = 0.3
rng = np.random.default_rng(42)

x_test = np.linspace(-3, 3, 200)
y_true_test = np.sin(x_test)

for degree in degrees:
    all_predictions = []
    all_test_mse = []

    X_test_poly = make_polynomial_features(x_test, degree)

    for _ in range(n_trials):
        X_train, y_train = generate_dataset(
            n_samples=n_train_samples,
            noise_std=noise_std,
            x_range=(-3, 3),
            rng=rng,
        )

        model = fit_polynomial_regression(X_train, y_train, degree)

        y_pred_test = model.predict(X_test_poly)
        all_predictions.append(y_pred_test)

        mse = mean_squared_error(y_true_test, y_pred_test)
        all_test_mse.append(mse)

    all_predictions = np.array(all_predictions)

    mean_prediction = all_predictions.mean(axis=0)

    bias2 = np.mean((mean_prediction - y_true_test) ** 2)
    variance = np.mean(np.var(all_predictions, axis=0))
    avg_test_error = np.mean(all_test_mse)

    print(
        f"Degree={degree:>2} | "
        f"Bias^2={bias2:.6f} | "
        f"Variance={variance:.6f} | "
        f"Test Error={avg_test_error:.6f}"
    )



