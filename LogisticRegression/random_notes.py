import numpy as np





# z = X @ w
# pred = sigmoid_function(z)

# Toy dataset
X = np.array([
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4],
])

y = np.array([0, 0, 1, 1])  # classification

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(y, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(
        y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
    )

w = np.zeros(X.shape[1])
lr = 0.1

for epoch in range(1000):
    z = X @ w
    y_pred = sigmoid(z)

    grad = (1 / len(y)) * X.T @ (y_pred - y)
    w -= lr * grad

    if epoch % 100 == 0:
        print(f"Loss: {loss(y, y_pred):.4f}")

print("Weights:", w)
print("Predictions:", sigmoid(X @ w))



