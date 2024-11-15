import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances

# Membaca dataset
file_path = "./clustering.csv"
data = pd.read_csv(file_path)

print("Data Awal:\n", data.head())

# Memilih fitur untuk clustering
features = data[['X1', 'X2', 'X3', 'X4']]

# Inisialisasi K-Means
num_clusters = 5
num_init = 10
kmeans = KMeans(n_clusters=num_clusters, n_init=num_init, random_state=42)

# Melakukan clustering
data['cluster'] = kmeans.fit_predict(features)

# Menampilkan hasil clustering
print("\nHasil Clustering:\n", data)

# Menampilkan centroid tiap klaster
centroids = kmeans.cluster_centers_
print("\nCentroid Tiap Klaster:")
for i, centroid in enumerate(centroids):
    print(f"Centroid Klaster {i}: {centroid}")

# Menyimpan hasil clustering ke file CSV
data.to_csv("./clustering_result.csv", index=False)
print("\nHasil clustering disimpan ke 'clustering_result.csv'")

# Visualisasi 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    data['X1'], data['X2'], data['X3'],
    c=data['cluster'], cmap='viridis', s=data['X4'], alpha=0.8
)
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

# Menghitung jarak Euclidean antara data dan centroid
euclidean_distances_matrix = euclidean_distances(features, centroids)

# Menampilkan jarak Euclidean antara setiap data dan centroid
print("\nJarak Euclidean antara data dan centroid:")
print(euclidean_distances_matrix)

# Menyimpan jarak Euclidean ke file CSV
euclidean_distances_df = pd.DataFrame(euclidean_distances_matrix, columns=[f'Cluster_{i}_Distance' for i in range(num_clusters)])
data_with_distances = pd.concat([data, euclidean_distances_df], axis=1)
data_with_distances.to_csv("./clustering_with_distances.csv", index=False)

print("\nJarak Euclidean disimpan ke 'clustering_with_distances.csv'")