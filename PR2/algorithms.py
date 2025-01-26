import numpy as np
from collections import Counter

def calculate_distance(point1, point2, metric="euclidean"):
    match metric:
        case "euclidean":
            return np.sqrt(np.sum((point1 - point2) ** 2))
        case "manhattan":
            return np.sum(np.abs(point1 - point2))
        case _:
            raise ValueError(f'Unsupported metric {metric}. Acceptable metrics = euclidean, manhattan')

class CustomKNNClassifier:
    def __init__(self, k = 3, metric="euclidean"):
        self.k = k
        self.metric = metric
    
    def fit(self, X, y):
        self.X_train  = X
        self.y_train = y

        return self
    
    def predict(self, X_test):
        distances = self._compute_distances(X_test)
        k_neighbour_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_neighbour_labels = self.y_train[k_neighbour_indices]
        # print(k_neighbour_labels)

        # y_pred = [self._find_most_common_label(neighbours) for neighbours in k_neighbour_labels]
        y_pred = [self._find_most_common_label(neighbours) for neighbours in k_neighbour_labels]
        return np.array(y_pred)

    def _compute_distances(self, X_test):
        n_test = X_test.shape[0]
        n_train = self.X_train.shape[0]

        distances = np.zeros((n_test, n_train))
        for i in range(n_test):
            for j in range(n_train):
                distances[i,j] = calculate_distance(point1=X_test[i], point2=self.X_train[j], metric=self.metric)
                # print(i,j, distances[i,j])

        return distances
    
    def _find_most_common_label(self, neighbours):
        label_counts = {}
        for label in neighbours:
            # label = label.item()
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