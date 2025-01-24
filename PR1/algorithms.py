# Foundations of Data Mining - Practical Task 1
# Version 2.1 (2024-10-27)
###############################################
# Template for a custom clustering library.
# Classes are partially compatible to related scikit-learn classes.
# Aside from check_array, do not import functions from scikit-learn, tensorflow, keras or related libraries!
# Do not change the signatures of the given functions or the class names!

import numpy as np
from sklearn.utils import check_array
from collections import Counter


def calculate_distance(point1, point2, metric="euclidean"):
    """calculate_distance

    Args:
        point1 (_type_): _description_
        point2 (_type_): _description_
        metric (str, optional): _description_. Defaults to "euclidean". Acceptable: euclidean, manhattan
    """
    if metric == 'euclidean':
        return np.sqrt(np.sum((point1 - point2) ** 2))
    elif metric == 'manhattan':
        return np.sum(np.abs(point1 - point2))
    else:
        raise ValueError(f'Unsupported metric: {metric}. Acceptable values = euclidean, manhattan')


# def euclidean_distance(point, centroid):
#     return np.sqrt(np.sum((point - centroid) ** 2))


class KDTreeNode:
    def __init__(self, index, left=None, right=None):
        self.index = index
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, data, metric='euclidean'):
        self.data = data
        self.root = self._build_tree(indices=np.arange(len(data)), depth=0)
        self.metric = metric

    def _build_tree(self, indices, depth):
        if len(indices) == 0:
            return None

        k = self.data.shape[1]
        axis = depth % k
        sorted_indices = indices[np.argsort(self.data[indices, axis])]
        median_index = len(sorted_indices) // 2

        # print(axis)
        # print(sorted_indices)
        # print(median_index)

        return KDTreeNode(
            index= sorted_indices[median_index],
            left=self._build_tree(sorted_indices[:median_index], depth+1),
            right=self._build_tree(sorted_indices[median_index + 1 : ],depth+1)
        )
    
    def query_radius(self, index, radius):
        results = []
        point = self.data[index]
        self._query_radius(node=self.root, point=point, radius=radius, depth=0, results=results)
        return results

    def _query_radius(self, node, point, radius, depth, results):
        if node is None:
            return

        k = self.data.shape[1]
        axis = depth % k
        node_point = self.data[node.index]

        distance = calculate_distance(point, node_point, metric=self.metric)

        if distance <= radius:
            results.append(node.index)
        
        diff = point[axis] - node_point[axis]
        if diff <= radius:
            self._query_radius(node=node.left, point=point, radius=radius, depth=depth+1, results=results)
        if diff >= -radius:
            self._query_radius(node=node.right, point=point, radius=radius, depth=depth+1, results=results)
    

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

        #Initial random cluster center selection
        self.cluster_centers_ = X[np.random.choice(X.shape[0],self.n_clusters,replace=False)]

        for iteration in range(self.max_iter):
            # Uncomment below line for debug purposes only.
            # print(f'Iteration {iteration}, Cluster Centers = {self.cluster_centers_}')
            self.clusters_ = [[] for _ in range(self.n_clusters)]
            interm_labels = []

            for point in X:
                distances = [calculate_distance(point,cluster_center,metric='euclidean') for cluster_center in self.cluster_centers_]
                cluster_index = np.argmin(distances)
                self.clusters_[cluster_index].append(point)
                interm_labels.append(cluster_index)
            
                # Uncomment below lines for debug purposes only.
                # print(f'Point = {point}, distances = {distances}, cluster_index = {cluster_index}')
            # print(f'clusters = {self.clusters_}')

            new_cluster_centers = []
            for cluster in self.clusters_:
                if cluster:
                    new_cluster_center = np.mean(cluster,axis=0)
                    new_cluster_centers.append(new_cluster_center)
                else:
                    new_cluster_centers.append(self.cluster_centers_(new_cluster_centers))
            
            
            new_cluster_centers = np.array(new_cluster_centers) # Converted to np array to compare the new and old cluster centers

            if np.allclose(self.cluster_centers_, new_cluster_centers):
                # Uncomment below line for debug purposes only.
                # print(f'Converged after {iteration + 1} iterations')
                break

            self.cluster_centers_ = new_cluster_centers
            self.labels_ = interm_labels
        
        self.clusters_ = [np.array(cluster) for cluster in self.clusters_]

        # Determination of labels:
        # self.labels_ = None  # TODO: Implement your solution here!
        self.labels_ = np.array(self.labels_)

        return self

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_


class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', use_kdtree=True):
        """
        Creates an instance of CustomDBSCAN.
        :param min_samples: Equivalent to minPts. Minimum amount of neighbors of a core object.
        :param eps: Short for epsilon. Radius of considered circle around a possible core object.
        :param metric: Used metric for measuring distances.
        :param use_kdtree: Defaults to True. If enabled, k-dimentional tree is used for neighbor calculations. Else Brute Force Approach is used,
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.use_kdtree = use_kdtree
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

        # Determination of labels:
        # self.labels_ = None  # TODO: Implement your solution here!

        n = X.shape[0]
        self.labels_ = np.full(n, -1) # Mark all the points as Noise
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0
        kdtree = None

        if self.use_kdtree:
            kdtree = KDTree(X, metric=self.metric)
        
        # Main loop
        for point_idx in range(n):
            # print('Exploring point at index ', point_idx,' Clusters so far : ', Counter(self.labels_))
            if visited[point_idx]:
                continue

            visited[point_idx] = True
            if self.use_kdtree:
                neighbours = kdtree.query_radius(index=point_idx, radius=self.eps)
            else:
                neighbours = self._region_query(X=X, point_idx=point_idx)
            # neighbours = self._region_query(X=X, point_idx=point_idx, kdtree=kdtree)

            if len(neighbours) < self.min_samples:
                self.labels_[point_idx] = -1

            else:
                self._expand_cluster(X=X, point_idx=point_idx, cluster_id=cluster_id, neighbours=neighbours, visited=visited, kdtree=kdtree)
                cluster_id += 1
 
        return self
    

    def _region_query(self, X, point_idx):
        neighbours = []
        for idx, point in enumerate(X):
            if calculate_distance(point1=X[point_idx], point2=point, metric=self.metric) <= self.eps:
                neighbours.append(idx)

        # if self.use_kdtree:
        #     neighbours = kdtree.query_radius(index=point_idx, radius=self.eps)
        # else:
        #     for idx, point in enumerate(X):
        #         if calculate_distance(point1=X[point_idx], point2=point, metric=self.metric) <= self.eps:
        #             neighbours.append(idx)
        
        return neighbours

    def _expand_cluster(self, X, point_idx, cluster_id, neighbours, visited, kdtree):
        self.labels_[point_idx] = cluster_id

        i = 0
        while i < len(neighbours):
            neighbour_idx = neighbours[i]

            if not visited[neighbour_idx]:
                visited[neighbour_idx] = True
                if self.use_kdtree:
                    new_neighbours = kdtree.query_radius(index=neighbour_idx, radius=self.eps)
                else:
                    new_neighbours = self._region_query(X,neighbour_idx)
                
                # new_neighbours = self._region_query(X=X, point_idx=point_idx, kdtree=kdtree)

                if len(new_neighbours) >= self.min_samples:
                    neighbours = np.append(neighbours, new_neighbours)
            
            if self.labels_[neighbour_idx] == -1:
                self.labels_[neighbour_idx] = cluster_id

            i += 1

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_
