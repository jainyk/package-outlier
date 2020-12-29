import scipy
import scipy.stats as ss
import numpy as np
import pandas as pd

def entry_point():
    pass

def iqr_threshold_method(scores, margin):
    q1 = np.percentile(scores, 25, interpolation='midpoint')
    q3 = np.percentile(scores, 75, interpolation='midpoint')
    iqr = q3-q1
    lower_range = q1 - (1.5 * iqr)
    upper_range = q3 + (1.5 * iqr)
    lower_range = lower_range - margin
    upper_range = upper_range + margin
    return lower_range, upper_range


def ZscoreOutlier(data, margin=0, axis=0, ddof=0, nan_policy='propagate'):
    """Returns numpy array with data points labelled as outliers
    Parameters
    ----------
    data : array_like
        An array like object containing the sample data.

    margin : int, default=0
        Margin of error

    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over
        the whole array `a`.

    ddof : int, optional
        Degrees of freedom correction in the calculation of the
        standard deviation. Default is 0.

    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'.

    """
    scores = ss.zscore(data, axis=axis, ddof=ddof, nan_policy=nan_policy)

    lower_range, upper_range = iqr_threshold_method(scores, margin)

    anomaly_points = []
    for i in range(len(scores)):
        if scores[i] < lower_range or scores[i] > upper_range:
            anomaly_points.append(data[i])

    return anomaly_points


def ModifiedZscoreOutlier(data, margin=0, consistency_correction=1.4826):
    """Returns numpy array with data points labelled as outliers
    Parameters
    ----------
    data : array_like
        An array like object containing the sample data.

    margin : int, default=0
        Margin of error

    consistency_correction : int, default=1.4826
        The consistency_correction factor converts the MAD to the standard deviation for a given
        distribution. The default value (1.4826) is the conversion factor if the underlying data
        is normally distributed

    """

    median = np.median(data)

    deviation_from_med = np.array(data) - median

    mad = np.median(np.abs(deviation_from_med))
    scores = deviation_from_med/(consistency_correction*mad)

    lower_range, upper_range = iqr_threshold_method(scores, margin)

    anomaly_points = []
    for i in range(len(scores)):
        if scores[i] < lower_range or scores[i] > upper_range:
            anomaly_points.append(data[i])

    return anomaly_points

if __name__=='__main__':
    arr = [8,5,7,8,11,13,4,9,10,7,6]
    arr = np.array(arr)
    res = ModifiedZscoreOutlier(arr, 1)
    print(res)
