import numpy as np
np.set_printoptions(suppress=True, precision=4)

import sys
import matplotlib.pyplot as plt
import sklearn.decomposition as decomp
import sklearn.linear_model as linear_model
import sklearn.datasets as sk_data
from sklearn.preprocessing import StandardScaler
import numpy.linalg as nla
import sklearn.svm as svm
import pandas as pd


def iqr_threshold_method(scores, margin):
    q1 = np.percentile(scores, 25, interpolation='midpoint')
    q3 = np.percentile(scores, 75, interpolation='midpoint')
    iqr = q3-q1
    lower_range = q1 - (1.5 * iqr)
    upper_range = q3 + (1.5 * iqr)
    lower_range = lower_range - margin
    upper_range = upper_range + margin
    return lower_range, upper_range


def get_projected_vectors(X, pca, ssX=None):
    if not ssX:
        ssX = StandardScaler().fit(X)
    centered_data = ssX.transform(X)
    reduced = pca.transform(centered_data)
    return ssX.inverse_transform(pca.inverse_transform(reduced))


def do_pca_anomaly_scores(obs,
    n_components):

    ssX = StandardScaler()
    centered_data = ssX.fit_transform(obs)

    pca = decomp.PCA(n_components = n_components)
    pca.fit(centered_data)

    projected_vectors = get_projected_vectors(obs, pca)
    return nla.norm(obs - projected_vectors, axis=1)


# currently the maximum value of pca score is considered as Anomaly
# hence only one anomaly is returned
# find a better way to get threshold value
# one could be determine outlier in pca score and use it as threshold
# similar issue with testing_svm_based
def PcaOutlier(data, n_components, margin=0):

    """Returns numpy array with data points labelled as outliers
    Parameters
    ----------
    data : numpy array like data_points

    n_components : int, float, None or str
        Number of components to keep.

    margin : int, default=0
        Margin of error
"""

    pca_score = do_pca_anomaly_scores(data, n_components)

    lower_range, upper_range = iqr_threshold_method(pca_score, margin)

    anomaly_points = []

    for i in range(len(pca_score)):
        if pca_score[i] < lower_range or pca_score[i] > upper_range:
            anomaly_points.append(data[i])

    return anomaly_points


if __name__=='__main__':
    pca_example = np.array([[-3, -1.5], [-2.5, -1.25], [-1.5, -0.75],
                            [-1, -0.5], [-0.5, -0.25], [0, 0], [0.5, 0.26],
                            [1, 0.5],  [1.5, 0.75], [2.5, 1.25], [3, 1.5]])

    res = PcaOutlier(pca_example, 1)
    print(res)
