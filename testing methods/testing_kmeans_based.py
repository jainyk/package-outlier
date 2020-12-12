# this also requires threshold determination method
# in this score can change if we run algo multiple times, there are 2 reasons:
# 1. different initialization points, hence different centroids may form
# 2. since different centroids is formed, score will vary

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as sk_data
import sklearn.neighbors as neighbors
import sys
from sklearn.cluster import KMeans

np.set_printoptions(suppress=True, precision=4)


def kmeans_outlier_detection(data, no_of_clusters):
    """Returns numpy array with data points labelled as outliers
    Parameters
    ----------
    data:
    no_of_clusters: numpy array like data_point
    score: numpy like data
    """
    km = KMeans(n_clusters=no_of_clusters).fit(data)
    centers = km.cluster_centers_[km.labels_]
    score = np.linalg.norm(data - centers, axis=1)
    return score


if __name__ == '__main__':
    blobs_X, cluster_labels = sk_data.make_blobs(
        centers=[[0, 0], [10, 10], [10, 0]])
    anomalies, _ = sk_data.make_blobs(
        centers=[[5, 5]], n_samples=5, cluster_std=3, random_state=42)

    data = np.concatenate([blobs_X, anomalies])
    res = kmeans_outlier_detection(data, 3)
    print(res)
