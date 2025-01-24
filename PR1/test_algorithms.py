# Foundations of Data Mining - Practical Task 1
# Version 2.1 (2024-10-27)
###############################################
# Template for a custom clustering library.
# Classes are partially compatible to related scikit-learn classes.
# Aside from check_array, do not import functions from scikit-learn, tensorflow, keras or related libraries!
# Do not change the signatures of the given functions or the class names!

import numpy as np
from sklearn.utils import check_array

def euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))


class CustomKMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        """
        Creates an instance of CustomKMeans.
        :param n_clusters: Amount of target clusters (=k).
        :param max_iter: Maximum amount of iterations before the fitting stops (optional).
        :param random_state: Initialization for randomizer (optional).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.clusters_ = None

    def fit(self, X: np.ndarray, y=None):
        """
        This is the main clustering method of the CustomKMeans class, which means that this is one of the methods you
        will have to complete/implement. The method performs the clustering on vectors given in X. It is important that
        this method saves the centroids in "self.cluster_centers_" and the labels (=mapping of vectors to clusters) in
        the "self.labels_" attribute! As long as it does this, you may change the content of this method completely
        and/or encapsulate the necessary mechanisms in additional functions.
        :param X: Array that contains the input feature vectors
        :param y: Unused
        :return: Returns the clustering object itself.
        """
        # Input validation:
        X = check_array(X, accept_sparse='csr')

        # Calculation of cluster centers:
        self.cluster_centers_ = None  # TODO: Implement your solution here!
        self.labels_ = None

        #Initial random cluster selection
        self.cluster_centers_ = X[np.random.choice(X.shape[0],self.n_clusters,replace=False)]

        for iteration in range(self.max_iter):
            # print(f'Iteration {iteration}, Cluster Centers = {self.cluster_centers_}')
            self.clusters_ = [[] for _ in range(self.n_clusters)]
            labels = []
            for point in X:
                distances = [euclidean_distance(point,cluster_center) for cluster_center in self.cluster_centers_]
                cluster_index = np.argmin(distances)
                self.clusters_[cluster_index].append(point)
                labels.append(cluster_index)
                # print(f'Point = {point}, distances = {distances}, cluster_index = {cluster_index}')
            # print(f'clusters = {self.clusters_}')

            new_cluster_centers = []
            for cluster in self.clusters_:
                if cluster:
                    new_cluster_center = np.mean(cluster,axis=0)
                    new_cluster_centers.append(new_cluster_center)
                else:
                    new_cluster_centers.append(self.cluster_centers_(new_cluster_centers))
            
            new_cluster_centers = np.array(new_cluster_centers)
            # self.cluster_centers_ = np.array(self.cluster_centers_)
            # print(f'Old Cluster Centers = {new_cluster_centers}. Shape = {self.cluster_centers_.shape}')
            # print(f'New Cluster Centers = {new_cluster_centers}. Shape = {new_cluster_centers.shape}')


            if np.allclose(self.cluster_centers_, new_cluster_centers):
                print(f'Converged after {iteration + 1} iterations')
                break

            self.cluster_centers_ = new_cluster_centers
            self.labels_ = labels
        
        self.clusters_ = [np.array(cluster) for cluster in self.clusters_]
        # print(self.clusters_)
        
        # Determination of labels:
        # self.labels_ = [] # TODO: Implement your solution here!
        # for point in X:
        #     for label, cluster in enumerate(self.clusters_):
        #         if point in cluster:
        #             self.labels_.append(label)

        self.labels_ = np.array(self.labels_)

        return self

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_


class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        """
        Creates an instance of CustomDBSCAN.
        :param min_samples: Equivalent to minPts. Minimum amount of neighbors of a core object.
        :param eps: Short for epsilon. Radius of considered circle around a possible core object.
        :param metric: Used metric for measuring distances.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None

    def fit(self, X: np.ndarray, y=None):
        """
        This is the main clustering method of the CustomDBSCAN class, which means that this is one of the methods you
        will have to complete/implement. The method performs the clustering on vectors given in X. It is important that
        this method saves the determined labels (=mapping of vectors to clusters) in the "self.labels_" attribute! As
        long as it does this, you may change the content of this method completely and/or encapsulate the necessary
        mechanisms in additional functions.
        :param X: Array that contains the input feature vectors
        :param y: Unused
        :return: Returns the clustering object itself.
        """
        # Input validation:
        X = check_array(X, accept_sparse='csr')

        """
            Notes:
            -------
            label info

            0 - Unvisited Point
            -1 - Noise Point
            n - assigned to cluster 'n'
        """

        # Determination of labels:
        self.labels_ = None  # TODO: Implement your solution here!
        self.labels_ = [0] * X.shape[0] 

        return self

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_
