## Project description
 This is pypi package for outlier detection

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/package-outlier)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/package-outlier)
![PyPI](https://img.shields.io/pypi/v/package-outlier?style=flat-square)
[![GitHub issues](https://img.shields.io/github/issues/amit22666/package-outlier)](https://github.com/amit22666/package-outlier/issues)
[![GitHub forks](https://img.shields.io/github/forks/amit22666/package-outlier?style=social)](https://github.com/amit22666/package-outlier/network)
[![GitHub stars](https://img.shields.io/github/stars/amit22666/package-outlier)](https://github.com/amit22666/package-outlier/stargazers)
[![GitHub license](https://img.shields.io/github/license/amit22666/package-outlier?style=flat-square)](https://github.com/amit22666/package-outlier/blob/main/LICENSE)
![PyPI - Downloads](https://img.shields.io/pypi/dm/package-outlier)







## Install
Read the online Installation instructions.

This software depends on NumPy and Scipy, Python packages for scientific computing. You must have them installed prior to installing package-outlier.

Install the latest version of package-outlier

```$ pip install package-outlier```

This will display a message and download if the module is not already installed. It will then install package-outlier and all its dependencies.

## Dependencies
```
     Python
     NumPy
     SciPy
     scikit-learn
     pandas
```


## How to call a function
```
      import package_outlier as po
      result = po.ZscoreOutlier(data)
      result = po.ModifiedZscoreOutlier(data)
      result = po.LocalOutlierFactorOutlier(data)
      result = po.DepthOutlier(data)
      result = po.KmeansOutlier(data)
      result = po.OdinOutlier(data)
      result = po.RegressionOutlier(data)
      result = po.SvmOutlier(data)
      result = po.PcaOutlier(data)
      result = po.KnnOutlier(data)
      result = po.AngleOutlier(data)
```
## NOTE: In all implementations we have used interquartile range based method to define the threshold value.
      The formula used for evaluation is as follows:
      lower_range = q1 - (1.5 * iqr)
      upper_range = q3 + (1.5 * iqr)
      lower_range = lower_range - margin
      upper_range = upper_range + margin

## Zscore based outlier detection
Zscore is a common method to detect anomaly in 1-D.
For a given data point zscore is calculated by:\
zscore = data_point - mean / std_dev

The function take data and threshold value as required argument and returns data points that are outliers.

## Modified zscore based outlier detection
Mean and standard deviation are themselves prone to outliers that's why we use median instead of mean and median absolute deviation instead of mean absolute deviation.\
For more info on median absolute deviation refer to https://en.wikipedia.org/wiki/Median_absolute_deviation.


## Angle based outlier detection
For a normal point the angle it makes with any other two data points varies a lot as you choose
different data points.
For an anomaly the angle
it makes with any other two data
points doesn’t vary much as you
choose different data points
Here we used cosθ to calculate angle between 2 vectors.

## Depth based outlier detection
 Outliers lie at the edge of the data space. According to this concept we organize the data in layers
in which each layer is labeled by its depth. The outermost layer is depth = 1, the next is
depth = 2 and so on. Finally outliers are those points with a depth below a predetermined threshold.
This implementation uses a convex hull to implement this depth based method. Convex hull is defined as the smallest convex set that contains the data.
This method is typically efficient only for two and three dimensional data. Outliers are points with a depth ≤ n.

## Linear regression based outlier detection
You should be familiar with linear regression in order to understand this method. In this vertical distance from straight line fit is used to score points.
Outliers are far from line i.e, the distance between regression fitted line and data point is far. A threshold value is calculated using these scores in order to label data point as outlier.
NOTE that linear regression in itself is sensitive to outliers

## PCA based outlier detection

You should be familiar with PCA in order to understand this method.
The principal components are linear combinations of the original features.
Usually few principal components matter since they accompanies most of the variance of the data and hence most of the data aligns along a lower-dimensional feature space.
Outliers are those points that don’t align with this subspace. Distance of the anomaly from the aligned data can be used as an anomaly score. Outlier itself can affect the modelling
hence it should be modelled on normal data point and then should be used to detect outliers.

## SVM based outlier detection
In this one class SVM is used for outlier detection. Basically the idea is data points lieing to one side of hyperplane is considered as normal
and other side as data points is labelled as outliers. Two key assumptions while applying it are:
1. Data provided all belong to normal class
   Since data may contain anomalies this results in a noisy model
2. The origin belongs to the anomaly class
   Rarely use data as is. Origin is that of kernel-based transformed data
NOTE:
1. The shape of the decision boundary is sensitive to the choice of kernel and
other tuning parameters of SVMs
2. Without deep knowledge of both the data and SVMs, it is easy to get poor
results
3. To address this issue, sampling of subsets of the data and averaging of scores
is recommended.


## KNN based outlier detection
The basic idea is anomalies are far away from neighboring points. In this for each point, distance is calculated to k nearest neighbors.
Now we can take either take arithematic mean or harmonic mean of the obtained KNN distances to set the threshold value and values
exceeding this limit is considered as outlier.
NOTE:
1. The value of k and scoring process affect the results
2. Choosing k requires judgment hence a range of values is used
3. It is a good idea to check the scoring process, if results vary wildly with the choice of distance metric and scoring threshold,
further examination of the data is recommended.

## ODIN based outlier detection
This method is considered to be the reverse of KNN. For each point it's KNN are considered which is called the indegree number of that data point.
Large indegree number means that instance is the neighbor of many points hence it is labelled as normal points and small indegree number means that instance is relatively isolated
hence it is termed as outlier.

## K means based outlier detection
We should be familiar with the working of k-means while diving to this method.
The basic idea is outliers are far away clusters (dense collections of points).
Now usually there are 3 types of distances to be considered like distance from cluster centroid,
distance from cluster edge and Mahalanobis distances to each cluster.
NOTE:
1. Choice of k affects the results
2. Initial choice of centroids can also affect results

## LOF based outlier detection
It is a density based method in which outliers are located in sparse regions. It defines outliers with respect to local region, Compares local density of query point with local density of neighbors
 and if the local density of the query point is much lower then it is labelled as outlier.
The process is as followed-
1. Define local region around query point by its k nearest neighbors (“query
neighbors”)
2. For far away query neighbors, use distance between query neighbor and
query point
– For close neighbors, use distance to the kth nearest neighbor of the query neighbor
3.  Average distances over all query neighbors is known as “average reachability distance”
    Local density = reciprocal of average reachability distance
    LOF = average local density of neighbors / local density of query point
   – LOF ≈ 1 similar density as neighbors
   – LOF < 1 higher density than neighbors (normal point)
   – LOF > 1 lower density than neighbors (outlier)
