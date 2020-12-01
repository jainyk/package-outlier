# Here we are having 3 options i.e, max value, mean value and harmonic_mean value to select condense distances from given k values
# but all of them will suffer if we have outlier in training datasets
# so this is way to go solution further research is needed
# but selecting threshold_value is still big challenge
# one method could be find standard deviation and if outlier_score are sorted compare whether each point is distance to each other less than standard deviation
# currently only outlier_score is calculated

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as sk_data
import sklearn.neighbors as neighbors
import sys
import statistics

np.set_printoptions(suppress=True, precision=4)


def max_knn_outlier_scores(obs, n_neighbors=1):
    nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(obs)
    dists, idx = nn.kneighbors()
    scores = dists[:,-1]
    return scores


def mean_knn_outlier_scores(obs, n_neighbors=1):
    nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(obs)
    dists, idx = nn.kneighbors()
    scores = []
    for i in range(0, dists.shape[0]):
        m = statistics.mean(dists[i])
        scores.append(m)
    return scores


def harmonic_mean_outlier_scores(obs, n_neighbors=1):
    nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(obs)
    dists, idx = nn.kneighbors()
    scores = []
    for i in range(0, dists.shape[0]):
        m = statistics.harmonic_mean(dists[i])
        scores.append(m)
    return scores


def threshold_value_predictor(data, n_neighbors, flag=1):
    if flag == 1:
        outlier_scores = max_knn_outlier_scores(data, n_neighbors)
    if flag == 2:
        outlier_scores = mean_knn_outlier_scores(data, n_neighbors)
    if flag == 3:
        outlier_scores = harmonic_mean_outlier_scores(data, n_neighbors)

#    threshold_value = max(outlier_scores)
    return outlier_scores


if __name__=='__main__':
    X1 = np.array([[0.9, 1], [0, 1], [1, 0], [0, 0], [0.5, 0.5], [0.2, 0.5], [1, 0.5]])
    X2 = np.array([[0.9, 1], [0, 1], [1, 0], [0, 0], [0.5, 0.5], [0.2, 0.5], [1, 0.5], [2, 2]])

    res = threshold_value_predictor(X2, 3, 2)
    print(res)
