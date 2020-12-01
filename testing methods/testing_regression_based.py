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


def regression_based_outlier(x_train, y_train, x_test, y_test, margin=0.01, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None):
    """Returns numpy array with data points labelled as outliers
    Parameters
    ----------
    x_train: numpy array like
    data points on which regression to be trained

    y_train: numpy array like
    lables of x_train

    x_test: numpy array like
    data points in which outliers to be determined

    y_test: numpy array like
    lables of x_test

    margin: int, default=0.01
    Margin of error while calculating threshold with threshold = max(train_residuals) + margin

    fit_intercept: bool, default=True
    Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).

    normalize: bool, default=False
    This parameter is ignored when fit_intercept is set to False. If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm. If you wish to standardize, please use sklearn.preprocessing.StandardScaler before calling fit on an estimator with normalize=False.

    copy_X: bool, default=True
    If True, X will be copied; else, it may be overwritten.

    n_jobs: int, default=None
    The number of jobs to use for the computation. This will only provide speedup for n_targets > 1 and sufficient large problems. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
"""
    lr_train = linear_model.LinearRegression(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs).fit(x_train, y_train)
    train_scores = (y_train - lr_train.predict(x_train))**2

    margin = margin
    threshold = max(train_scores) + margin
    test_scores = (y_test - lr_train.predict(x_test))**2

    anomaly_points = []
    for i in range(len(test_scores)):
        if test_scores[i] > threshold:
            anomaly_points.append(x_test[i])
    return np.stack(anomaly_points)
