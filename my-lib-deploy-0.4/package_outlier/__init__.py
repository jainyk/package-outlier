import scipy
import scipy.stats as ss
import numpy as np
import pandas as pd
import random
import math
import sys
import sklearn.decomposition as decomp
import sklearn.linear_model as linear_model
import sklearn.datasets as sk_data
from sklearn.preprocessing import StandardScaler
import numpy.linalg as nla
import sklearn.datasets as sk_data
import sklearn.svm as svm
import sklearn.neighbors as neighbors
import statistics
from sklearn.cluster import KMeans
import datetime
from scipy.spatial import ConvexHull
np.set_printoptions(suppress=True, precision=4)


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


def SvmOutlier(data, margin=0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1):
    """Returns numpy array with data points labelled as outliers
    Parameters
    ----------
    data: numpy array like data_points

    margin : int, default=0
        Margin of error

    kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
        Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to precompute the kernel matrix.

    degree: int, default=3
        Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.

    gamma: {‘scale’, ‘auto’} or float, default=’scale’
        Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.

        if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,

        if ‘auto’, uses 1 / n_features.

    coef0: float, default=0.0
        Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.

    tol: float, default=1e-3
        Tolerance for stopping criterion.

    nu: float, default=0.5
        An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. Should be in the interval (0, 1]. By default 0.5 will be taken.

    shrinking: bool, default=True
        Whether to use the shrinking heuristic. See the User Guide.

    cache_size: float, default=200
        Specify the size of the kernel cache (in MB).

    verbose: bool, default=False
        Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.

    max_iter: int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.

    all_data: numpy like data
    """

    oc_svm = svm.OneClassSVM(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, nu=nu, shrinking=shrinking, cache_size=cache_size, verbose=verbose, max_iter=max_iter).fit(data)
    scores = oc_svm.decision_function(data).flatten()

    # Find the largest score and use it to normalize the scores
    max_score = np.max(np.abs(scores))

    # scores from oc_svm use "negative is anomaly"
    # To follow our previous convention
    # we multiply by -1 and divide by the maximum score to get scores
    # in the range [-1, 1] with positive values indicating anomalies
    svm_scores = - scores / max_score

    #mapping = {}
    lower_range, upper_range = iqr_threshold_method(svm_scores, margin)

    outlier_scores = []
    data_points = []

    for i in range(0,len(svm_scores)):
    #    mapping[svm_scores[i]] = i
        if svm_scores[i] < lower_range or svm_scores[i] > upper_range:
            outlier_scores.append(svm_scores[i])
            data_points.append(data[i])

    return np.stack(data_points)


def RegressionOutlier(x_train, y_train,  margin=0, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None):
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


def get_projected_vectors(X, pca, ssX=None):
    if not ssX:
        ssX = StandardScaler().fit(X)
    centered_data = ssX.transform(X)
    reduced = pca.transform(centered_data)
    return ssX.inverse_transform(pca.inverse_transform(reduced))


def do_pca_anomaly_scores(obs,
    n_components):

    ssX = StandardScaler()
    centered_data = ssX.fit_transform(obs)

    pca = decomp.PCA(n_components = n_components)
    pca.fit(centered_data)

    projected_vectors = get_projected_vectors(obs, pca)
    return nla.norm(obs - projected_vectors, axis=1)


# currently the maximum value of pca score is considered as Anomaly
# hence only one anomaly is returned
# find a better way to get threshold value
# one could be determine outlier in pca score and use it as threshold
# similar issue with testing_svm_based
def PcaOutlier(data, n_components, margin=0):

    """Returns numpy array with data points labelled as outliers
    Parameters
    ----------
    data : numpy array like data_points

    n_components : int, float, None or str
        Number of components to keep.

    margin : int, default=0
        Margin of error
"""

    pca_score = do_pca_anomaly_scores(data, n_components)

    lower_range, upper_range = iqr_threshold_method(pca_score, margin)

    anomaly_points = []

    for i in range(len(pca_score)):
        if pca_score[i] < lower_range or pca_score[i] > upper_range:
            anomaly_points.append(data[i])

    return anomaly_points


def OdinOutlier(
    data,
    margin=0,
    n_neighbors=5,
    radius=1.0,
    algorithm='auto',
    leaf_size=30,
    metric='minkowski',
    p=2,
    metric_params=None,
    n_jobs=None
):
    """Returns numpy array with data points labelled as outliers
    Parameters
    ----------
    data : numpy array like data_points

    margin : int, default=0
        Margin of error

    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    radius : float, default=1.0
        Range of parameter space to use by default for :meth:`radius_neighbors`
        queries.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors

    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    metric : str or callable, default='minkowski'
        the distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. See the documentation of :class:`DistanceMetric` for a
        list of available metrics.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`,
        in which case only "nonzero" elements may be considered neighbors.

    p : int, default=2
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
    """

    nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors, radius=radius, algorithm=algorithm, leaf_size=leaf_size, metric=metric, p=p, metric_params=metric_params, n_jobs=n_jobs)

    nn.fit(data)
    graph = nn.kneighbors_graph()
    indegree = graph.sum(axis=0)  # sparse matrix

    # smaller indegree means more of an anomaly
    # simple conversion to [0,1] so larger score is more of anomaly
    scores = (indegree.max() - indegree) / indegree.max()
    scores = np.array(scores)[0]

    lower_range, upper_range = iqr_threshold_method(scores, margin)

    anomaly_points = []
    for i in range(len(scores)):
        if scores[i] < lower_range or scores[i] > upper_range:
            anomaly_points.append(data[i])

    return anomaly_points


def KmeansOutlier(
    data,
    margin=0,
    n_clusters=8,
    init='k-means++',
    n_init=10,
    max_iter=300,
    tol=0.0001,
    precompute_distances='deprecated',
    verbose=0,
    random_state=None,
    copy_x=True,
    n_jobs='deprecated',
    algorithm='auto',
):
    """Returns numpy array with data points labelled as outliers
    Parameters
    ----------
    data : numpy array like data_points

    margin : int, default=0
        Margin of error

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random', ndarray, callable}, default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    precompute_distances : {'auto', True, False}, default='auto'
        Precompute distances (faster but takes more memory).

        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.

        True : always precompute distances.

        False : never precompute distances.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    n_jobs : int, default=None
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    algorithm : {"auto", "full", "elkan"}, default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient on data with well-defined
        clusters, by using the triangle inequality. However it's more memory
        intensive due to the allocation of an extra array of shape
        (n_samples, n_clusters).

    """

    km = KMeans(n_clusters=n_clusters,init=init,n_init=n_init,
    max_iter=max_iter,tol=tol,precompute_distances=precompute_distances,verbose=verbose,
    random_state=random_state,copy_x=copy_x,n_jobs=n_jobs,algorithm=algorithm
).fit(data)

    centers = km.cluster_centers_[km.labels_]
    scores = np.linalg.norm(data - centers, axis=1)

    lower_range, upper_range = iqr_threshold_method(scores, margin)

    anomaly_points = []
    for i in range(len(scores)):
        if scores[i] < lower_range or scores[i] > upper_range:
            anomaly_points.append(data[i])

    return anomaly_points


def depth_calculator(data, dict, count):
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


def DepthOutlier(points, threshold):
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


def LocalOutlierFactorOutlier(data,
    margin=0,
    n_neighbors=20,
    algorithm='auto',
    leaf_size=30,
    metric='minkowski',
    p=2,
    metric_params=None,
    contamination='auto',
    novelty=False,
    n_jobs=None):

    """Returns numpy array with data points labelled as outliers

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of neighbors to use by default for :meth:`kneighbors` queries.
        If n_neighbors is larger than the number of samples provided,
        all samples will be used.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to :class:`BallTree` or :class:`KDTree`. This can
        affect the speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.

    metric : str or callable, default='minkowski'
        metric used for the distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a sparse matrix, in which case only "nonzero"
        elements may be considered neighbors.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics:
        https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

    p : int, default=2
        Parameter for the Minkowski metric from
        :func:`sklearn.metrics.pairwise.pairwise_distances`. When p = 1, this
        is equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. When fitting this is used to define the
        threshold on the scores of the samples.

        - if 'auto', the threshold is determined as in the
          original paper,
        - if a float, the contamination should be in the range [0, 0.5].

        .. versionchanged:: 0.22
           The default value of ``contamination`` changed from 0.1
           to ``'auto'``.

    novelty : bool, default=False
        By default, LocalOutlierFactor is only meant to be used for outlier
        detection (novelty=False). Set novelty to True if you want to use
        LocalOutlierFactor for novelty detection. In this case be aware that
        that you should only use predict, decision_function and score_samples
        on new unseen data and not on the training set.

        .. versionadded:: 0.20

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    """
    lof = neighbors.LocalOutlierFactor(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, metric=metric, p=p, metric_params=metric_params, contamination=contamination, novelty=novelty, n_jobs=n_jobs)

    lof.fit(data)

    scores = -lof.negative_outlier_factor_
    scores = list(scores)

    lower_range, upper_range = iqr_threshold_method(scores, margin)

    outlier_points = []

    for i in range(len(scores)):
        if scores[i] < lower_range or scores[i] > upper_range:
            outlier_points.append(data[i])

    return outlier_points


def KnnOutlier(data,
    margin=0,
    n_neighbors=5,
    radius=1.0,
    algorithm='auto',
    leaf_size=30,
    metric='minkowski',
    p=2,
    metric_params=None,
    n_jobs=None):

    """Returns numpy array with data points labelled as outliers
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    radius : float, default=1.0
        Range of parameter space to use by default for :meth:`radius_neighbors`
        queries.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    metric : str or callable, default='minkowski'
        the distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. See the documentation of :class:`DistanceMetric` for a
        list of available metrics.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`,
        in which case only "nonzero" elements may be considered neighbors.

    p : int, default=2
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
"""
    nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors, radius=radius, algorithm=algorithm, leaf_size=leaf_size, metric=metric, p=p, metric_params=metric_params, n_jobs=n_jobs)

    nn.fit(data)

    dists, idx = nn.kneighbors()

    scores = []
    for i in range(0, dists.shape[0]):
        m = statistics.mean(dists[i])
        scores.append(m)

    lower_range, upper_range = iqr_threshold_method(scores, margin)

    outlier_points = []

    for i in range(len(scores)):
        if scores[i] < lower_range or scores[i] > upper_range:
            outlier_points.append(data[i])

    return outlier_points


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
