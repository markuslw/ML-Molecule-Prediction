import os
# CPU max
os.environ["LOKY_MAX_CPU_COUNT"] = '24'
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['MKL_NUM_THREADS'] = '24'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

"""
    Load the data using pandas and create numpy array
"""
def load_data(data_path):
    data_path = 'data/frey-faces.csv'
    data = pd.read_csv(filepath_or_buffer=data_path, delim_whitespace=True, comment='#', header=None, dtype=np.float64)
    
    """
        The data is a 2D array of shape (1965, 560). Each row represents a 28x20 image.
    """
    images = data.to_numpy()
    images = images.reshape(-1, 28, 20)

    images_flattened = images.reshape(images.shape[0], -1)
    return images, images_flattened

"""
    Perform k-means clustering on the images and find the closest_dp images closest to each centroid.
"""
def k_means(k, images_flattened, closest_dp=5):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(images_flattened)
    centroids = kmeans.cluster_centers_

    """
        Calculate the closest images to each centroid using the euclidean distance
        with linalg.norm(). Use argsort() to get the index of the closest images.
    """
    closest_images = []
    for centroid in centroids:
        distances = np.linalg.norm(images_flattened - centroid, axis=1)
        closest_indices = np.argsort(distances)[:closest_dp]
        closest_images.append(closest_indices)
    centroids_images = centroids.reshape(k, 28, 20)

    print(closest_images)
    
    return centroids_images, closest_images

"""
    Plot the centroids and the closest images in a grid.
"""
def plot_images(images, closest_images, centroids_images, k, closest_dp=5):
    _, axes = plt.subplots(k, closest_dp+1, figsize=(10, 2 * k))
    for i in range(k):
        # Display the centroid
        axes[i, 0].imshow(centroids_images[i], cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'Centroid {i+1}')

        # Display the 5 closest images
        for j, index in enumerate(closest_images[i]):
            axes[i, j + 1].imshow(images[index], cmap='gray')
            axes[i, j + 1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    k_values = [3, 6, 10]
    data_path = 'data/frey-faces.csv'

    images, images_flattended = load_data(data_path=data_path)

    for k in k_values:
        centroid_images, closest_images = k_means(k=k, images_flattened=images_flattended)
        plot_images(k=k, images=images, closest_images=closest_images, centroids_images=centroid_images)
