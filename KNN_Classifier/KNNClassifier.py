import numpy as np
from collections import Counter


class KNNClassifier:
    def __init__(self, k_neighbors=3):
        self.k = k_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Stores the training data (lazy learning)."""
        self.X_train = X_train
        self.y_train = y_train
        pass

    def _euclidean_distance(self, point1, point2):
        """Helper function for distance calculation."""
        return np.sqrt(np.sum((point1 - point2)**2))

    def _predict_single(self, test_point):
        """Predicts the class for a single test point."""
        distances = [self._euclidean_distance(test_point, train_point) for train_point in self.X_train]
        # Get the indices that would sort the distances
        k_nearest_indices = np.argsort(distances)[:self.k]
        # Get the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
        # Return the most frequent label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X_test):
        """Predicts classes for an array of test points."""
        predictions = [self._predict_single(test_point) for test_point in X_test]
        return np.array(predictions)

