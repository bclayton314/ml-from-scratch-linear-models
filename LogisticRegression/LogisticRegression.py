import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid_function(self, z):
        return 1 / (1 + np.exp(-z))

    #cross entropy loss
    def compute_loss():
        pass

    def _compute_loss(self, y_true, y_pred, samples):
        """Calculates the binary cross-entropy loss."""
        # Add a small epsilon (1e-15) to prevent log(0) errors
        epsilon = 1e-15
        loss = -(1/samples) * np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
        return loss

    # Gradient
    # gradient = (1 / n) * X.T @ (y_pred - y)

    def fit(self, X, y):
        """Trains the model using gradient descent."""
        num_samples, num_features = X.shape
        # Initialize weights and bias to zeros
        self.weights = np.zeros((num_features, 1))
        self.bias = 0
        y = y.reshape(-1, 1) # Ensure y is a column vector

        # Gradient Descent loop
        for _ in range(self.num_iterations):
            # Calculate the linear model output (z)
            z = np.dot(X, self.weights) + self.bias
            # Apply the sigmoid function to get predicted probabilities
            y_pred = self._sigmoid(z)

            # Calculate gradients
            # dw (gradient of loss w.r.t. weights)
            dw = (1/num_samples) * np.dot(X.T, (y_pred - y))
            # db (gradient of loss w.r.t. bias)
            db = (1/num_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self

    def predict(self, X):
        """Makes binary predictions (0 or 1) based on a 0.5 threshold."""
        z = np.dot(X, self.weights) + self.bias
        probabilities = self._sigmoid(z)
        # Convert probabilities to class labels
        return (probabilities >= 0.5).astype(int)



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



