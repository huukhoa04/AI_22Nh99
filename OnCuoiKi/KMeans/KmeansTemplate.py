import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
class KMeans:
    def __init__(self, k: int, max_iters: int = 100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def fit(self, X: np.ndarray) -> None:
        # Randomly initialize centroids
        n_samples, n_features = X.shape
        idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            old_centroids = self.centroids.copy()
            distances = self._calculate_distances(X)
            self.labels = np.argmin(distances, axis=1)

            # Update centroids
            for i in range(self.k):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    self.centroids[i] = np.mean(cluster_points, axis=0)

            # Check convergence
            if np.all(old_centroids == self.centroids):
                break

    def _calculate_distances(self, X: np.ndarray) -> np.ndarray:
        distances = np.zeros((X.shape[0], self.k))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
        return distances

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = self._calculate_distances(X)
        return np.argmin(distances, axis=1)

def load_and_preprocess_data(filepath: str) -> tuple[np.ndarray, StandardScaler]:
    # Load data
    df = pd.read_csv(filepath)
    
    # Assuming all columns are numerical features
    X = df.values
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

def plot_clusters(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> None:
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidth=3)
    plt.colorbar(scatter)
    plt.title('K-means Clustering Results')
    plt.show()

def main():
    # Load and preprocess data
    filepath = "data.csv"  # Replace with your CSV file path
    X_scaled, scaler = load_and_preprocess_data(filepath)

    # Initialize and fit K-means
    k = 3  # Number of clusters
    kmeans = KMeans(k=k)
    kmeans.fit(X_scaled)

    # Get cluster assignments
    labels = kmeans.labels

    # Plot results (if data is 2D)
    if X_scaled.shape[1] == 2:
        plot_clusters(X_scaled, labels, kmeans.centroids)

    # Print cluster information
    for i in range(k):
        cluster_size = np.sum(labels == i)
        print(f"Cluster {i} size: {cluster_size}")
if __name__ == "__main__":
    main()
