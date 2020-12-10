# Work in progress
# Same doubt as in testing_pca_based.py that how to determine threshold from svm score
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


def SvmOutlier(data, margin=0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1):
    """Returns numpy array with data points labelled as outliers
    Parameters
    ----------
    data: numpy array like data_points

    margin : int, default=0
        Margin of error

    kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
        Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to precompute the kernel matrix.

    degree: int, default=3
        Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.

    gamma: {‘scale’, ‘auto’} or float, default=’scale’
        Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.

        if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,

        if ‘auto’, uses 1 / n_features.

    coef0: float, default=0.0
        Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.

    tol: float, default=1e-3
        Tolerance for stopping criterion.

    nu: float, default=0.5
        An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. Should be in the interval (0, 1]. By default 0.5 will be taken.

    shrinking: bool, default=True
        Whether to use the shrinking heuristic. See the User Guide.

    cache_size: float, default=200
        Specify the size of the kernel cache (in MB).

    verbose: bool, default=False
        Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.

    max_iter: int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.

    all_data: numpy like data
    """

    oc_svm = svm.OneClassSVM(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, nu=nu, shrinking=shrinking, cache_size=cache_size, verbose=verbose, max_iter=max_iter).fit(data)
    scores = oc_svm.decision_function(data).flatten()

    # Find the largest score and use it to normalize the scores
    max_score = np.max(np.abs(scores))

    # scores from oc_svm use "negative is anomaly"
    # To follow our previous convention
    # we multiply by -1 and divide by the maximum score to get scores
    # in the range [-1, 1] with positive values indicating anomalies
    svm_scores = - scores / max_score

    #mapping = {}
    lower_range, upper_range = iqr_threshold_method(svm_scores, margin)

    outlier_scores = []
    data_points = []

    for i in range(0,len(svm_scores)):
    #    mapping[svm_scores[i]] = i
        if svm_scores[i] < lower_range or svm_scores[i] > upper_range:
            outlier_scores.append(svm_scores[i])
            data_points.append(data[i])

    return np.stack(data_points)



if __name__=='__main__':
    blobs_X, y = sk_data.make_blobs(centers=[[0,0], [10,10]])

    spike_1 = np.array([[6.0,6.0]]) # Anomaly 1
    spike_2 = np.array([[0.0,10]])  # Anomaly 2
    cluster_data = np.concatenate([blobs_X, spike_1, spike_2])
    res = SvmOutlier(cluster_data, 0.01)
    print(res)
