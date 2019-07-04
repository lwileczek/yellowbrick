# yellowbrick.cluster.dbcv
# Implements Density-Based Clustering Validation score for DBSCAN
#
# Author:   Luke Wileczek <lwileczek@protonmail.com>
# Created:  Sat May 4 10:20:38 2019
#
# Copyright (C) 2019 District Data Labs
# For license information, see LICENSE.txt
#
# ID: dbcv.py

"""
Implimenting Density-Based Clustering Validation
http://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import scipy.sparse as ss
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from .base import ClusteringScoreVisualizer
from ..style.palettes import LINE_COLOR
from ..exceptions import YellowbrickValueError, YellowbrickWarning

## Packages for export
__all__ = [
    "dbcv"
]

##########################################################################
## Functions
##########################################################################

"""
Start witht the eight definitions in the original paper
"""

def core_distance(X, labels, dist_func="euclidean"):
    """
    The inverse of the density of each object with respect to all other objects
    inside its cluster

    Parametes
    ---------
    X:  (array) Matrix of floats, The distances are calculated for each
        point in a cluster, not against all other points. 

    labels: (array) list of labels indicating to which cluster each record
        belongs e.g. sklearn.cluster.DBSCAN.fit().labels_

    dist_func: (string) string indicating the distance function to be used in

    OUTPUT:
        core_distances - (array) list of the core distance value for each point
        with respect to each other point in its cluster.  If it is a noise
        point, the value is zero. 

    assertions: 
    input matrix is a numpy array:  
    https://stackoverflow.com/questions/12569452/how-to-identify-numpy-types-in-python
    """

    n_rows, n_cols = X.shape
    core_distances = np.zeros(n_rows)
    if (len(set(labels)) - (1 if -1 in labels else 0)) == 0:
        return core_distances
    clusters = set(labels)
    if -1 in clusters:
        clusters.remove(-1)  # -1 stands for noise
    for clsr in clusters:
        bln = labels == clsr
        cluster = X[bln]
        n = sum(bln) - 1 
        core_dists = np.zeros(len(cluster))
        cluster_dists = cdist(cluster, cluster, dist_func)
        with np.errstate(divide='ignore'):

            cluster_dists = cluster_dists**(-n_cols)
        cluster_dists[cluster_dists == np.inf] = 0
        core_distances[bln] = (np.sum(cluster_dists, axis=1)/n) ** (-1/n_cols)

    return core_distances


def mutual_reachability(X, core_dists, dist_func="euclidean"):
    """
    The mutual reachability between two objects i.e. distance between two
    points or records.  Take the distance between the two points and select the
    maximum value between that distance and the core distance of both points.

    Max(d(p0, p1), p0_coredist, p1_coredist)

    Parameters
    ----------

    X:  (array) data matrix

    core_distances:  (array) The core distances for each row/object 

    dist_func:  (string) what distance metric to use in cdist


    OUTPUT:
        reachability_graph - A sparse lower triangular matrix of the 
        reachability distance between objects i and j
    """

    dist_matrix = cdist(X, X, dist_func)
    n_row, _ = dist_matrix.shape
    # n_row, n_col = X.shape

    # compute sparse matrix format
    length = n_row * (n_row-1) // 2
    dimensions = np.zeros((2, length), dtype=int)
    entries = np.zeros((3, length))
    for n in range(1, n_row):
        start = (n * (n+1) // 2)-n
        end = ((n+1) * (n+2) // 2)-n-1
        dimensions[1, start:end] = list(range(n))  # cols
        dimensions[0, start:end] = n               # rows
        entries[0, start:end] = dist_matrix[n, :n]
        #entries[0, start:end] = np.reshape(
        #    cdist(X[:n, :],X[n, :].reshape(1, n_col)), (n,))  
        entries[1, start:end] = core_dists[n]
        entries[2, start:end] = core_dists[range(n)]

    reachability = np.max(entries, axis = 0)
    reachability_graph = ss.csr_matrix((reachability, 
        (dimensions[0, :], dimensions[1, :])), shape=(n_row, n_row))

    return reachability_graph


def dbcv(X, labels, distance_function="euclidean"):
    """
    Density-Based Cluster Validation

    Calculate the DBCV which is a calculation on in the interval [-1, 1] where
    larger numbers are better.  This validaiton compares the maximum distance
    between between points in a cluster against the minimum distance to another
    clusters. 

    Origional Paper:
    http://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf

    Parameters
    ----------
    X:  (array) Matrix of floats, The distances are calculated for each
        point in a cluster, not against all other points. 

    labels: (array) list of labels indicating to which cluster each record
        belongs e.g. sklearn.cluster.DBSCAN.fit().labels_

    dist_func: (string) string indicating the distance function to be used in

    OUTPUT: 
       dbcv - (float) Validation number in the interval [-1, 1] 
    """
    assert isinstance(X, np.ndarray), "X must be a numpy array, np.ndarray"
    assert len(X) == len(labels), ("Must have a label for each point, -1 for"
        "noise")

    clusters = list(set(labels))
    cardinality = len(X)
    if -1 in clusters:
        clusters.remove(-1)

    distances = core_distance(X, labels, dist_func=distance_function)
    graph = mutual_reachability(X, distances, dist_func=distance_function)
    spanning_trees = []
    density_seperation = [ [ ] ] * len(clusters)  # list to hold all the seps
    for i,c in enumerate(clusters):
        idx = np.where(labels == c)[0]
        spanning_trees.append(minimum_spanning_tree(
            graph[idx, idx.reshape(len(idx), 1)]))
        seperation = []
        for cj in clusters:  # definitions 6
            if cj == c:
                continue
            idx_j = np.where(labels == cj)[0]
            sep = np.min((graph + graph.T)[idx, idx_j.reshape(len(idx_j), 1)])
            seperation.append(sep)
        density_seperation[i] = seperation
    density_sparseness = [np.max(tree) for tree in spanning_trees]  # DSC 
    #print("seperation:", density_seperation)
    #print("sparseness:", density_sparseness)

    # definition 7 validity index of a  cluster
    validity = [ (min(dspc) - dsc) / max(min(dspc), dsc) for dspc, dsc 
        in zip(density_seperation, density_sparseness)]

    # definition 8
    dbcv = sum([np.sum(labels == c)/cardinality * vc for c,vc in
        enumerate(validity)])

    return dbcv

