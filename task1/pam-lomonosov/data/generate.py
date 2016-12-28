#!/usr/bin/env python3

import sys

import numpy as np
from scipy.spatial.distance import *
import matplotlib.pyplot as plt

def break_intervals(n, n_intervals):
    start, size = 0, int(n/n_intervals)
    intervals = [size]

    for i in range(1, n_intervals):
        start += size
        size = int((n - start) / (n_intervals - i))
        intervals.append(size)

    return intervals

def main(argc=len(sys.argv), argv=sys.argv):
    np.random.seed(12345)

    X = []

    n_clusters, n_points = int(argv[1]), int(argv[2])
    n_points_in_cluster = break_intervals(n_points, n_clusters)

    # Generate cluster centroids
    cluster_centroids = np.random.randint(-10000, 10001, size=(n_clusters, 2))
    dx = dy = min(pdist(cluster_centroids))

    for i, centroid in enumerate(cluster_centroids):
        np.random.seed(i)
        points = np.random.multivariate_normal(centroid, cov=[[dx**2, 0], [0, dy**2]], size=n_points_in_cluster[i])
        X.append(points)
        # plt.plot(points[:, 0], points[:, 1], 'ro')

    X = np.vstack(X)
    X_distances = cdist(X, X)
    np.savetxt('{}_{}_points.csv'.format(n_clusters, n_points), X, fmt='%1.4f', delimiter=',', header='')
    np.savetxt('{}_{}_distances.csv'.format(n_clusters, n_points), X_distances, fmt='%1.4f', delimiter=',', header='')

    # plt.plot(cluster_centroids[:, 0], cluster_centroids[:, 1], 'bo')
    # plt.show()

if __name__ == "__main__":
    main()