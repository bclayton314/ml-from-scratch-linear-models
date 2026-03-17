import numpy as np

X = np.array([
    [1, 1],
    [1, 2],
    [1, 3]
], dtype=float)

y = np.array([2, 3, 4], dtype=float)

# Closed-Form Solution
# w = (X^T * X)^-1 * (X ** T) * y

w = np.linalg.inv(X.T @ X) @ (X.T) @ y

print(w)
