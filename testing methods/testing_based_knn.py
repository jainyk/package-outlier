#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Here we are having 3 options i.e, max value, mean value and harmonic_mean value to select condense distances from given k values
# but all of them will suffer if we have outlier in training datasets
# so this is way to go solution further research is needed
# but selecting threshold_value is still big challenge
# one method could be find standard deviation and if outlier_score are sorted compare whether each point is distance to each other less than standard deviation
# currently only outlier_score is calculated

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as sk_data
import sklearn.neighbors as neighbors
import sys
import statistics

np.set_printoptions(suppress=True, precision=4)


# In[2]:


def iqr_threshold_method(scores, margin):
    q1 = np.percentile(scores, 25, interpolation='midpoint')
    q3 = np.percentile(scores, 75, interpolation='midpoint')
    iqr = q3-q1
    lower_range = q1 - (1.5 * iqr)
    upper_range = q3 + (1.5 * iqr)
    lower_range = lower_range - margin
    upper_range = upper_range + margin
    return lower_range, upper_range


# In[3]:


def max_knn_outlier_scores(
    obs, 
    n_neighbors=1,
    margin = 0
    ):
    """Returns numpy array with data points labelled as outliers
    
    Parameters
    ----------
    
    obs:numpy array like data_points
    
        
    algorithm : {'n_neighbors'}, 
        Algorithm used to capture the idea of similarity
    
    n_neighbours: int , default=1
        Your chosen number of neighbors
    
    margin : int, default=0
        Margin of error
    
    dists: array like data_points
        distance between the query point and the current point
    
    scores: numpy like data
    
    """
    

    nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(obs)
    dists, idx = nn.kneighbors()
    scores = dists[:,-1]
#     print(scores)
#     print(type(scores))
#     print(scores.shape)
    scores = scores.reshape(-1,1)

    lower_range, upper_range = iqr_threshold_method(scores , margin)
    
#     print(lower_range)
#     print(upper_range)

    outlier_scores = []
    data_points = []

    for i in range(0,len(scores)):
    #    mapping[svm_scores[i]] = i
        if scores[i] < lower_range or scores[i] > upper_range:
            outlier_scores.append(scores[i])
            data_points.append(obs[i])

#     print(obs)
    return outlier_scores


# In[4]:



if __name__=='__main__':
    X1 = np.array([[0.9, 1], [0, 1], [1, 0], [0, 0], [0.5, 0.5], [0.2, 0.5], [1, 0.5]])
    X2 = np.array([[0.9, 1], [0, 1], [1, 0], [0, 0], [0.5, 0.5], [0.2, 0.5], [1, 0.5], [2, 2]])

    res = max_knn_outlier_scores(X1,0)
    print(res)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




