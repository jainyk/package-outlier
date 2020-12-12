# only data point and score (both in tuple) are returned by function
# threshold is still not selected 
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as sk_data
import sklearn.neighbors as neighbors
import sys
import statistics


def do_odin_outlier_scores(obs, n_neighbors=3):

     """Returns numpy array with data points labelled as outliers
    Parameters
    ----------
    data:
    no_of_clusters: numpy array like data_point
    score: numpy like data
    """


    nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors).fit(obs)
    graph = nn.kneighbors_graph()
    indegree = graph.sum(axis=0)  # sparse matrix

    # smaller indegree means more of an anomaly
    # simple conversion to [0,1] so larger score is more of anomaly
    scores = (indegree.max() - indegree) / indegree.max()
    return np.array(scores)[0]  # convert to array


def print_ranked_scores(obs, scores):
     """Returns numpy array with data points labelled as outliers
    Parameters
    ----------
    data:
    no_of_clusters: numpy array like data_point
    score: numpy like data
    """


    scores_and_obs = sorted(zip(scores, obs), key=lambda t: t[0], reverse=True)
    return scores_and_obs
#    scores_points_dict = {}
#    print(scores_and_obs)
#    for index, score_ob in enumerate(scores_and_obs):
#        score, point = score_ob
#        scores_points_dict[score] = point
#        scores.append(score)
#        points.append(point)
#    return scores_points_dict

if __name__=='__main__':
    X2 = np.array([[0.9, 1], [0, 1], [1, 0], [0, 0], [0.5, 0.5], [0.2, 0.5], [1, 0.5], [2, 2]])

    scores_odin= do_odin_outlier_scores(X2)
    #print(scores_odin)

    arg1 = print_ranked_scores(X2, scores_odin)
    print(arg1)
