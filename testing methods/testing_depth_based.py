import sys
import datetime
import scipy
import scipy.stats as ss
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.spatial import ConvexHull


def depth_calculator(data, dict, count):

 """Returns numpy array with data points labelled as outliers
    Parameters
    ----------
    data : numpy array like data_points
    
    algorithm : {'n_neighbors , contamination'}, 
        Algorithm used to compute the depth
    
    
    margin : int, default=0
        Margin of error
        
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    """
    next_data = []
    if len(data) < 3: # in 2D, at least three points are needed to make a convex hull
        for index, item in enumerate(data):
            dict[tuple(item)] = count # assign depth to remaining points
    else:
        hull = ConvexHull(data) # Construct convex hull
        for index, item in enumerate(data):
            if item in data[hull.vertices]:
                dict[tuple(item)] = count # assign depth to points on hull
            else:
                dict[tuple(item)] = count + 1.0 # assign depth+1 to points not on hull
                next_data.append(item)
        new_data =np.asarray(next_data) # create new data file of points not on hull
        depth_calculator(new_data, dict, count + 1.0) # repeat on new data file
    return dict # returns dictionary with data_point as tuple and assigned value equal to depth


def depth_based_anomaly_detection(points, threshold):
    """Returns numpy array with data points labelled as outliers
    Parameters
    ----------
    points: numpy array like data_point
    threshold: depth value below which points are considered as anomaly"""
    depth_dict = {} # empty dictionary for depth_calculator
    depth_calculator(points, depth_dict, 1.0) # initial hull has depth 1.0
    anomaly_points = [data_point for data_point, depth in depth_dict.items() if depth < threshold] # data_point with depth less than given threshold is returned as anomaly point
    anomaly_points = np.array(anomaly_points)
    return anomaly_points
