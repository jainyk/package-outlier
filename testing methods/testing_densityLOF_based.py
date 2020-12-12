#!/usr/bin/env python
# coding: utf-8

# In[13]:


import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as sk_data
import sklearn.neighbors as neighbors
import sys
from sklearn.cluster import KMeans


# In[14]:


np.set_printoptions(suppress=True, precision=4)

import sklearn.neighbors as neighbors


# In[15]:


def iqr_threshold_method(scores, margin):
    q1 = np.percentile(scores, 25, interpolation='midpoint')
    q3 = np.percentile(scores, 75, interpolation='midpoint')
    iqr = q3-q1
    lower_range = q1 - (1.5 * iqr)
    upper_range = q3 + (1.5 * iqr)
    lower_range = lower_range - margin
    upper_range = upper_range + margin
    return lower_range, upper_range


# In[16]:


def localOutlierFactor_based_outlier(data, n_neighbors, contamination='auto'):

    """Returns numpy array with data points labelled as outliers
    Parameters
    ----------
    data : numpy array like data_points
    
    algorithm : {'n_neighbors , contamination'}, 
        Algorithm used to compute the LocalOutlierFactor
    
    contamination: ?, default='auto'
    
    margin : int, default=0
        Margin of error
        
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    """
    lof = neighbors.LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    lof.fit(data)
    sk_lof = -lof.negative_outlier_factor_
    print_ranked_scores(X2, sk_lof)
    
    lower_range, upper_range = iqr_threshold_method(sk_lof,0)
    
#     print(lower_range)
#     print(upper_range)

    outlier_scores = []
    data_points = []

    for i in range(0,len(sk_lof)):
    #    mapping[svm_scores[i]] = i
        if sk_lof[i] < lower_range or sk_lof[i] > upper_range:
            outlier_scores.append(sk_lof[i])
            data_points.append(data[i])
    
    return outlier_scores


# In[17]:


if __name__=='__main__':
    X = np.array([[0.9, 1], [0, 1], [1, 0], [0, 0], [0.5, 0.5], [0.2, 0.5], [1, 0.5], [2, 2]])
    X2 = np.concatenate([X, [[1.9, 2.0]]])
    res = localOutlierFactor_based_outlier(X2, 3)
    print(res)


# In[ ]:




