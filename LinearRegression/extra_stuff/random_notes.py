
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import os
import pandas as pd
#import matplotlib.pyplot as plt

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data.csv')

data = pd.read_csv(data_path)

"""
linear_regression_plot.py

"""


# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data.csv')

#data = pd.read_csv(data_path)



"""
linear_regression_plot.py

Reads a CSV with integer columns x and y, fits a simple linear regression
(using closed-form least squares), prints the fitted line, and plots:
- scatter of data points
- best-fit regression line

Usage (from the folder containing data.csv):
    python linear_regression_plot.py

If your CSV file is not named data.csv, change CSV_PATH below.
"""

CSV_PATH = data_path  # expects columns: x, y


def load_xy(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find '{csv_path}'.\n"
            f"Current working directory: {os.getcwd()}\n"
            f"Tip: put data.csv next to this script, or set CSV_PATH to an absolute path."
        )

    df = pd.read_csv(csv_path)

    # Basic column validation
    for col in ("x", "y"):
        if col not in df.columns:
            raise ValueError(f"CSV must contain columns 'x' and 'y'. Found: {list(df.columns)}")

    # Ensure numeric
    x = pd.to_numeric(df["x"], errors="raise").to_numpy(dtype=float)
    y = pd.to_numeric(df["y"], errors="raise").to_numpy(dtype=float)

    if x.size == 0:
        raise ValueError("CSV contains no rows.")

    return x, y


def fit_linear_regression_closed_form(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Fits y = m*x + b via least squares closed-form:
        m = cov(x,y)/var(x)
        b = mean(y) - m*mean(x)
    """
    x_mean = x.mean()
    y_mean = y.mean()

    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        raise ValueError("All x values are identical; cannot fit a line.")

    m = np.sum((x - x_mean) * (y - y_mean)) / denom
    b = y_mean - m * x_mean
    return float(m), float(b)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 1.0


def main() -> None:
    try:
        x, y = load_xy(CSV_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    m, b = fit_linear_regression_closed_form(x, y)
    y_hat = m * x + b
    r2 = r2_score(y, y_hat)

    print(f"Fitted line: y = {m:.6f} * x + {b:.6f}")
    print(f"R^2: {r2:.6f}")

    # Plot
    plt.figure()
    plt.scatter(x, y, label="Data")
    # Plot the line over a sorted x range
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = m * x_line + b
    plt.plot(x_line, y_line, label="Best-fit line")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Linear Regression Fit (R^2 = {r2:.4f})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()



# def gradient_descent(m_current, b_current, points, l_rate):
#     m_grad = 0
#     b_grad = 0

#     n = len(points)

#     for i in range(n):
#         x = points.iloc[i].x
#         y = points.iloc[i].y

#         m_grad += -2/n * x * (y - (m_current * x + b_current))
#         b_grad += -2/n * (y - (m_current * x + b_current))

#     m = m_current - m_grad * l_rate
#     b = b_current - b_grad * l_rate

#     return m, b

# m = 0
# b = 0
# l_rate = 0.0001
# epochs = 200

# for i in range(epochs):
#     if i % 50 == 0:
#         print(f"Epoch: {i}")
#     m, b = gradient_descent(m, b, data, l_rate)

# print(m, b)

# plt.scatter(data.x, data.y, color="black")
# plt.plot(list(range(0, 100)), [m * x + b for x in range(0, 100)], color="red")
# plt.show()


# # Hook this up to class
# # -unit_tests
# # -do math beforehand tomorrow
