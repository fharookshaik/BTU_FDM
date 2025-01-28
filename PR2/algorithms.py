# Foundations of Data Mining - Practical Task 2
###############################################

import numpy as np



def calculate_distance(point1, point2, metric="euclidean"):
    """Calculates the distance betwwen two points using the specified metric.

    Args:
        point1 : Coordinates of point 1
        point2 : Coordinates of point 2
        metric (str, optional): Distance metric to use. Options are:
            - "euclidean" (default)
            - "manhattan
            - "chebyshev"

    Raises:
        ValueError: If an unsupported metric is specified.

    Returns:
        float: Returns the calculated distance between point1 and point2
    """
    match metric:
        case "euclidean":
            return np.sqrt(np.sum((point1 - point2) ** 2))
        case "manhattan":
            return np.sum(np.abs(point1 - point2))
        case "chebyshev":
            return max(abs(a - b) for a,b in zip(point1, point2))
        case _:
            raise ValueError(f'Unsupported metric {metric}. Acceptable metrics = euclidean, manhattan, chebyshev')



class CustomKNNClassifier:
    """
    A custom implementation of the K-Nearest Neighbors (KNN) classifier.

    Parameters:
        k (int): The number of nearest neighbors to consider for classification (default is 3).
        metric (str): The distance metric to use for calculating distances. 
                      Supported values: 'euclidean', 'manhattan', 'chebyshev'.

    Methods:
        fit(X, y): Stores the training data.
        predict(X_test): Predicts labels for the given test data.
    """


    def __init__(self, k = 3, metric="euclidean"):
        """
        Initializes the CustomKNNClassifier.

        Args:
            k (int): Number of neighbors.
            metric (str): Distance metric ('euclidean', 'manhattan', 'chebyshev').
        """
        self.k = k
        self.metric = metric
    
    def fit(self, X, y):
        """
        Stores the training data.

        Args:
            X (array-like): Feature matrix for training data (n_samples, n_features).
            y (array-like): Labels for the training data (n_samples).

        Returns:
            self: Returns the instance of the classifier.
        """
        self.X_train  = X
        self.y_train = y

        return self
    
    def predict(self, X_test):
        """
        Predicts the class labels for the given test data.

        Args:
            X_test (array-like): Feature matrix for test data (n_samples, n_features).

        Returns:
            np.array: Predicted labels for the test data.
        """
        distances = self._compute_distances(X_test)
        k_neighbour_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_neighbour_labels = self.y_train[k_neighbour_indices]

        y_pred = [self._find_most_common_label(neighbours) for neighbours in k_neighbour_labels]
        return np.array(y_pred)

    def _compute_distances(self, X_test):
        """
        Computes the distance between each test instance and all training instances.

        Args:
            X_test (array-like): Test feature matrix (n_samples, n_features).

        Returns:
            np.array: Matrix of distances (n_test_samples, n_train_samples).
        """
        n_test = X_test.shape[0]
        n_train = self.X_train.shape[0]

        distances = np.zeros((n_test, n_train))
        for i in range(n_test):
            for j in range(n_train):
                distances[i,j] = calculate_distance(point1=X_test[i], point2=self.X_train[j], metric=self.metric)

        return distances
    
    def _find_most_common_label(self, neighbours):
        """
        Finds the most common label among the neighbors.

        Args:
            neighbours (array-like): Labels of the nearest neighbors.

        Returns:
            int/str: The most common label.
        """
        label_counts = {}
        for label in neighbours:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
            
        most_common_label = None
        max_count = 0

        for label, count in label_counts.items():
            if count > max_count:
                max_count = count
                most_common_label = label
        
        return most_common_label