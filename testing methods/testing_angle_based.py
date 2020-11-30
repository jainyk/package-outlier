import scipy
import scipy.stats as ss
import numpy as np
import matplotlib
import pandas as pd
'''
import random
# Given points A, B and C, this function returns the acute angle between vectors AB and AC
# using the dot product between these vectors
np.random.seed(16) # include a seed for reproducibility

# generate the normal data
normal_mean = np.array([1.0, 2.0])
normal_covariance = np.array([[0.2, 0.0], [0.0, 0.1]])
normal_data = np.random.multivariate_normal(normal_mean, normal_covariance, 100)

# generate the anomalous data
anomaly_mean = np.array([6.0, 8.0])
anomaly_covariance = np.array([[2.0, 0.0], [0.0, 4.0]])
anomaly_data = np.random.multivariate_normal(anomaly_mean, anomaly_covariance, 10)

final_data = np.concatenate((normal_data, anomaly_data), axis=0)
all_data = np.concatenate((normal_data, anomaly_data), axis=0)


def angle(point1, point2, point3):
    v21 = np.subtract(point2, point1)
    v31 = np.subtract(point3, point1)
    dot_product = (v21*v31).sum()
    normalization = np.linalg.norm(v21)*np.linalg.norm(v31)
    print(type(normalization))
    print(normalization)
    acute_angle = np.arccos(dot_product/normalization)
    return acute_angle
'''

def angle(point1, point2, point3):
    v21 = np.subtract(point2, point1)
    v31 = np.subtract(point3, point1)
    dot_product = (v21*v31).sum()
    normalization = (np.linalg.norm(v21)**2) * (np.linalg.norm(v31)**2)
    acute_angle = np.arccos(dot_product/normalization)
    return acute_angle


def eval_angle_point(point, data):
    angles_data = []
    for index_b, b in enumerate(data):
        if (np.array_equal(b, point)):
            continue
        # ensure point c comes later in array that point b
        # so we don't double count points
        for c in data[index_b + 1:]:
            if (np.array_equal(c, point)) or (np.array_equal(c, b)):
                continue
            angles_data.append(angle(point, b, c))
    return angles_data



def angle_based_anomaly_detection(points, all_data, threshold):
    """Returns numpy array with data points labelled as outliers
    Parameters
    ----------
    points: numpy array like data_point
    all_data: numpy like data
    threshold: value below which points are considered as anomaly"""

    df_ = pd.DataFrame(columns=['point','angle variance'])
    for index2, item2 in enumerate(points):
        df_.loc[index2] = [item2, np.var(eval_angle_point(item2, all_data))]

    outliers = df_[df_['angle variance'] < threshold ]
    outliers = np.stack(outliers['point'].to_numpy())
    return outliers


#------------------------------------------------------------------------------
