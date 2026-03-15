
import numpy as np


"""

"""

# 1. Dot Product Practice 
# vec1 = np.array([1, 2, 3])
# vec2 = np.array([4, 5, 6])

# dot_prod1 = np.sum(np.multiply(vec1, vec2))
# dot_prod2 = 0

# for i in range(len(vec1)):
#     dot_prod2 += vec1[i] * vec2[i]

# print(dot_prod1)
# print(dot_prod2)


# 2. Matrix-Vector Multiplication
# matrix1 = np.array([
#     [1, 2],
#     [3, 4],
#     [5, 6]
# ])

# w = np.array([10, 20])

# y2 = matrix1 @ w

# print(y2)


# 3. Add an Intercept Column
# X = np.array([
#     [2],
#     [4],
#     [6]
# ])

# intercept_vec = np.array([[1], [1], [1]])

# y = np.hstack((X, intercept_vec))
# print(np.shape(X))
# print(np.shape(intercept_vec))

# print(y)


# 4. Compute Predictions
# X = np.array([
#     [1, 2],
#     [1, 4],
#     [1, 6]
# ])

# w = np.array([3, 2])

# y_bar = np.dot(X, w)

# print(y_bar)


# 5. Compute Residuals
# y_true = np.array([8, 10, 14])
# y_pred = np.array([7, 11, 15])

# residuals = y_true - y_pred
# print(residuals)


# 6. Compute Mean Squared Error
# y_true = np.array([8, 10, 14])
# y_pred = np.array([7, 11, 15])

# MSE = np.mean((y_true - y_pred) ** 2)
# print(MSE)


# 7. Compute the Closed-Form Solution



