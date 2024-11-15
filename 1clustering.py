import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Membaca dataset
file_path = "./clustering.csv"
data = pd.read_csv(file_path)

print("Data Awal:\n", data.head())

features = data[['X1', 'X2', 'X3', 'X4']]

num_clusters = 5
num_init = 10
kmeans = KMeans(n_clusters=num_clusters, n_init=num_init, random_state=42)

# Clustering
data['cluster'] = kmeans.fit_predict(features)

print("\nHasil Clustering:\n", data)

# Menampilkan centroid tiap klaster
centroids = kmeans.cluster_centers_
print("\nCentroid Tiap Klaster:")
for i, centroid in enumerate(centroids):
    print(f"Centroid Klaster {i}: {centroid}")

# Saving file to csv
data.to_csv("./clustering_result.csv", index=False)
print("\nHasil clustering disimpan ke 'clustering_result.csv'")

# Visualisasi
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot data
scatter = ax.scatter(
    data['X1'], data['X2'], data['X3'],
    c=data['cluster'], cmap='viridis', s=data['X4'], alpha=0.8
)

# Centroid
ax.scatter(
    centroids[:, 0], centroids[:, 1], centroids[:, 2],
    c='red', marker='x', s=200, label='Centroids'
)

ax.set_title("Clustering Result (3D Plot)")
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.show()