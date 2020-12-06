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


def regression_based_outlier(x_train, y_train,  margin=0, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None):
    """Returns numpy array with data points labelled as outliers
    Parameters
    ----------
    x_train: numpy array like
        data points on which regression to be trained

    y_train: numpy array like
        lables of x_train

    margin: int, default=0
        Margin of error

    fit_intercept: bool, default=True
        Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).

    normalize: bool, default=False
        This parameter is ignored when fit_intercept is set to False. If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm. If you wish to standardize, please use sklearn.preprocessing.StandardScaler before calling fit on an estimator with normalize=False.

    copy_X: bool, default=True
        If True, X will be copied; else, it may be overwritten.

    n_jobs: int, default=None
        The number of jobs to use for the computation. This will only provide speedup for n_targets > 1 and sufficient large problems. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. 
"""
    lr_train = linear_model.LinearRegression(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs).fit(x_train, y_train)

    train_scores = (y_train - lr_train.predict(x_train))**2

    lower_range, upper_range = iqr_threshold_method(train_scores, margin)

    anomaly_points = []

    for i in range(len(train_scores)):
        if train_scores[i] < lower_range or train_scores[i] > upper_range:
            anomaly_points.append([x_train[i], y_train[i]])
    return anomaly_points


if __name__=='__main__':

    exam_data1 = np.array([[1, 2, 3, 4, 5],
                    [57, 70, 76, 84, 91]]).T
    x_train = np.array([1,2,3,4,5]).reshape(-1,1)
    x_test = np.array([57,70,99,84,91]).reshape(-1,1)
    res = regression_based_outlier(x_train, x_test, 0.5)
    print(res)
