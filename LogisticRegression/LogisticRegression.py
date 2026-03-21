import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid_function(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_true, y_pred, samples):
        """Calculates the binary cross-entropy loss."""
        epsilon = 1e-15
        loss = -(1/samples) * np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
        return loss

    def fit(self, X, y):
        """Trains the model using gradient descent."""
        num_samples, num_features = X.shape
        # Initialize weights and bias to zeros
        self.weights = np.zeros((num_features, 1))
        self.bias = 0
        y = y.reshape(-1, 1) 

        # Gradient Descent loop
        for _ in range(self.num_iterations):
            # Calculate the linear model output (z) and apply the sigmoid function
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid_function(z)

            # Calculate gradients
            dw = (1/num_samples) * np.dot(X.T, (y_pred - y))
            db = (1/num_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self

    def predict(self, X):
        """Makes binary predictions (0 or 1) based on a 0.5 threshold."""
        z = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid_function(z)
        # Convert probabilities to class labels
        return (probabilities >= 0.5).astype(int)

    def score(self, X, y_true):
        """Calculates the accuracy of the model."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true.reshape(-1, 1))

