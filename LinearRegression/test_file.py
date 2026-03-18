import numpy as np
#from LinearRegression_ClosedF import LinearRegression
from LinearRegression_GradD import LinearRegressionGD


X = np.array([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
])

y = np.array([3.0, 5.0, 7.0, 9.0])  # y = 1 + 2x

#model = LinearRegression(fit_intercept=True, method="pinv")
model = LinearRegressionGD(
    fit_intercept=True,
    learn_rate=0.05,
    n_iters=5000,
)

model.fit(X, y)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("Predictions:", model.predict(X))
print("MSE:", model.mse(X, y))
print("R^2:", model.r_squared(X, y))