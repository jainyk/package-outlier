import scipy
import scipy.stats as ss
import numpy as np
import matplotlib
import pandas as pd
import random
import math


def iqr_threshold_method(scores, margin):
    q1 = np.percentile(scores, 25, interpolation='midpoint')
    q3 = np.percentile(scores, 75, interpolation='midpoint')
    iqr = q3-q1
    lower_range = q1 - (1.5 * iqr)
    upper_range = q3 + (1.5 * iqr)
    lower_range = lower_range - margin
    upper_range = upper_range + margin
    return lower_range, upper_range


def angle(point1, point2, point3):
    v21 = np.subtract(point2, point1)
    v31 = np.subtract(point3, point1)
    dot_product = (v21*v31).sum()
    normalization = np.linalg.norm(v21)*np.linalg.norm(v31)
    acute_angle = np.arccos(dot_product/normalization)
    return acute_angle


def eval_angle_point(point, data):
    angles_data = []
    for index_b, b in enumerate(data):
        if (np.array_equal(b, point)):
            continue

        for c in data[index_b + 1:]:
            if (np.array_equal(c, point)) or (np.array_equal(c, b)):
                continue
            angles_data.append(angle(point, b, c))
    return angles_data



def AngleOutlier(data, margin=0):
    """Returns numpy array with data points labelled as outliers
    Parameters
    ----------
    data: numpy 2d array like data points

    margin: int, default=0
        Margin of error
    """
    no_of_data_point = data.shape[0]
    variance_of_each_datapoint = []

    for i in range(0, no_of_data_point):
        point = data[i]
        temp = eval_angle_point(point, data)
        variance_of_each_datapoint.append(np.var(temp))

    lower_range, upper_range = iqr_threshold_method(variance_of_each_datapoint, margin)

    outlier_points = []

    for i in range(0, no_of_data_point):
        if variance_of_each_datapoint[i] < lower_range or variance_of_each_datapoint[i] > upper_range:
            outlier_points.append(data[i])

    return outlier_points, lower_range, upper_range, variance_of_each_datapoint


if __name__=='__main__':
    np.random.seed(16)

    normal_mean = np.array([1.0, 2.0])
    normal_covariance = np.array([[0.2, 0.0], [0.0, 0.1]])
    normal_data = np.random.multivariate_normal(normal_mean, normal_covariance, 10)

    anomaly_mean = np.array([6.0, 8.0])
    anomaly_covariance = np.array([[2.0, 0.0], [0.0, 4.0]])
    anomaly_data = np.random.multivariate_normal(anomaly_mean, anomaly_covariance, 10)
    all_data = np.concatenate((normal_data, anomaly_data), axis=0)
    print(all_data)
    print(all_data.shape)
#    point = all_data[0]
    #print(point)
    #res = eval_angle_point(point, all_data)
    res = AngleOutlier(all_data)
    print(res)
    #print(res)
