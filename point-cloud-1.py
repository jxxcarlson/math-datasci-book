import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# Create synthetic data with obvious clusters and some noise
def create_sample_data(n_samples=300):
    # Generate main clusters
    X, _ = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0,
                      random_state=42)
    
    # Add some noise
    noise = np.random.uniform(low=-10, high=10, size=(50, 2))
    X = np.vstack([X, noise])
    
    return X

# Function to plot clusters
def plot_clusters(X, labels, title):
    plt.figure(figsize=(10, 7))
    
    # Plot points colored by cluster
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:  # Noise points
            color = 'gray'
        
        mask = labels == label
        plt.scatter(X[mask, 0], X[mask, 1], c=[color], 
                   label=f'Cluster {label}' if label != -1 else 'Noise')
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Generate sample data
    X = create_sample_data()
    
    # Apply DBSCAN
    eps = 1.5        # Maximum distance between two samples to be considered neighbors
    min_samples = 5  # Minimum number of samples in a neighborhood to form a core point
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X)
    
    # Count number of clusters (excluding noise)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f'Number of clusters: {n_clusters}')
    print(f'Number of noise points: {n_noise}')
    
    # Plot results
    plot_clusters(X, cluster_labels, 
                 f'DBSCAN Clustering\n(eps={eps}, min_samples={min_samples})')
    