import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')



class KmeanCluster(object):
    def __init__(self, k, data, features ,max_iter = 10):
        self.cluster = k
        self.iter = max_iter
        self.data = np.array(pd.read_csv(data)[features])
        self.centroids, self.idx = self.find_k_means(self.data, self.cluster, self.iter)
        
    def initialize_K_centroids(self, X, K):
        m,n = X.shape
        k_rand = np.ones((K, n))
        k_rand = X[np.random.choice(range(len(X)), K, replace=False),:]
        return k_rand 
    
    def find_closest_centroids(self, X, centroids):
        m = len(X)
        c = np.zeros(m)
        for i in range(m):
            distances = np.linalg.norm(X[i] - centroids, axis=1)
            c[i] = np.argmin(distances)
        return c

    def compute_means(self, X, idx, K):
        m, n = X.shape
        centroids = np.zeros((K, n))
        for k in range(K):
            points_belong_k = X[np.where(idx == k)]
            centroids[k] = np.mean(points_belong_k, axis=0,)
        return centroids
    
    def find_k_means(self, X, K, max_iters=10):
        _, n = X.shape
        centroids = self.initialize_K_centroids(X, K) 
        centroid_history = np.zeros((max_iters, K, n))
        for i in range(max_iters):
            idx = self.find_closest_centroids(X, centroids)
            centroids = self.compute_means(X, idx, K)
        
        return centroids, idx 
    
kmean = KmeanCluster(6, 'Countries-exercise.csv', ["Longitude","Latitude"], max_iter = 100)
print(kmean.centroids)
print(kmean.idx)
# Plotting the final centroids
plt.figure(figsize=(10, 6))
plt.scatter(kmean.data[:, 0], kmean.data[:, 1], c=kmean.idx, s=50, cmap='viridis')
plt.scatter(kmean.centroids[:, 0], kmean.centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('K-Means Clustering')
plt.show()