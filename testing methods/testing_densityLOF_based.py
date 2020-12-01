import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as sk_data
import sklearn.neighbors as neighbors
import sys
from sklearn.cluster import KMeans

np.set_printoptions(suppress=True, precision=4)

import sklearn.neighbors as neighbors


def print_ranked_scores(obs, scores):
    scores_and_obs = sorted(zip(scores, obs), key=lambda t: t[0], reverse=True)
    print('Rank  Point\t\tScore')
    print('------------------------------')
    for index, score_ob in enumerate(scores_and_obs):
        score, point = score_ob
        print(f'{index+1:3d}.  {point}\t\t{score:6.4f}')


def localOutlierFactor_based_outlier(data, n_neighbors, contamination='auto'):
    lof = neighbors.LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    lof.fit(data)
    sk_lof = -lof.negative_outlier_factor_
    print_ranked_scores(X2, sk_lof)


if __name__=='__main__':
    X = np.array([[0.9, 1], [0, 1], [1, 0], [0, 0], [0.5, 0.5], [0.2, 0.5], [1, 0.5], [2, 2]])
    X2 = np.concatenate([X, [[1.9, 2.0]]])
    localOutlierFactor_based_outlier(X2, 3)
