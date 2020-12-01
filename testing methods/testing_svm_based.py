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


def do_svm_anomaly_scores(obs, threshold):

    oc_svm = svm.OneClassSVM(gamma='auto').fit(obs)
    scores = oc_svm.decision_function(obs).flatten()

    # Find the largest score and use it to normalize the scores
    max_score = np.max(np.abs(scores))

    # scores from oc_svm use "negative is anomaly"
    # To follow our previous convention
    # we multiply by -1 and divide by the maximum score to get scores
    # in the range [-1, 1] with positive values indicating anomalies
    svm_scores = - scores / max_score
    outliers_scores =  [x for x in svm_scores if x > threshold]
    dic1 = {}
    outlier_data = []
    for i in range(0, len(svm_scores)):
        dic1[svm_scores[i]] = obs[i]
    for x in outliers_scores:
        outlier_data.append(dic1[x])
    return outlier_data


if __name__=='__main__':
    blobs_X, y = sk_data.make_blobs(centers=[[0,0], [10,10]])

    spike_1 = np.array([[6.0,6.0]]) # Anomaly 1
    spike_2 = np.array([[0.0,10]])  # Anomaly 2
    cluster_data = np.concatenate([blobs_X, spike_1, spike_2])
    res = do_svm_anomaly_scores(cluster_data, 0.85)
    print(res)
