import scipy
import scipy.stats as ss
import numpy as np
import pandas as pd

def entry_point():
    pass


def zscore_and_anomaly_detection(data, threshold, flag=1, axis=0, ddof=0, nan_policy='propagate'):
# Calculate zscore using scipy
    zscore = ss.zscore(data, axis, ddof, nan_policy)

# Calculating data i.e greater than given threshold
    if flag == 1:
        mask = zscore > threshold
    if flag == 0:
        mask = zscore < threshold
    return zscore, data[mask]


def modified_zscore_and_anomaly_detection(data, threshold, flag=1, consistency_correction=1.4826):
    """
    Returns the modified z score and Median Absolute Deviation (MAD) from the scores in data.
    The consistency_correction factor converts the MAD to the standard deviation for a given
    distribution. The default value (1.4826) is the conversion factor if the underlying data
    is normally distributed
    """
    median = np.median(data)

    deviation_from_med = np.array(data) - median

    mad = np.median(np.abs(deviation_from_med))
    mod_zscore = deviation_from_med/(consistency_correction*mad)

# calculating data i.e greater than given threshold
    if flag == 1:
        mask = mod_zscore > threshold
    if flag == 0:
        mask = mod_zscore < threshold

    return mod_zscore, mad, data[mask]
