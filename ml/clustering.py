import numpy as np
import random
from pycontrol.data_science import measure
from pycontrol import params



def get_clusters(X, n_samples, n_features, n_clusters, init):
    if init == params.kmeansRandom:
        index_clusters = random.sample(range(0,n_samples), n_clusters)
        clusters = X[index_clusters]
    elif init == params.kmeanspp:
        # 随机选取第一个簇中心
        # Randomly select the center of the first cluster
        clusters = np.zeros(shape=(1, n_features))
        index_cluster = random.sample(range(0,n_samples), 1)
        cluster = X[index_cluster]
        clusters[0] = cluster

        while len(clusters) < n_clusters:
            dist = measure.euclidean_dist(X, clusters)
            min_dist = np.min(dist, axis=1)
            argsort = np.argsort(min_dist)
            sum_dist = np.sum(min_dist)

            r = np.random.uniform(0.8, 0.9)
            sum_dist *= r
            for i in range(n_samples):
                index = argsort[i]
                sum_dist -= min_dist[index]
                if sum_dist < 0:
                    clusters = np.append(clusters, X[index:index+1, :], axis=0)
                    break

    return clusters


def kmeans(X, n_clusters, init=params.kmeanspp, max_iter=300):
    n_samples, n_features = X.shape

    # 选择初始簇中心点
    # Select initial cluster center point
    # TODO 其他初始化簇新方法
    clusters = get_clusters(X, n_samples, n_features, n_clusters, init)

    y_temp = np.zeros(shape=(n_samples,), dtype=np.int)
    for iter in range(max_iter):
        dist = measure.euclidean_dist(X, clusters)
        y = np.argmin(dist, axis=1)
        if (y == y_temp).all():
            break
        y_temp = y

        for i in range(n_clusters):
            x = X[y == i]
            clusters[i] = measure.cluster_center(x)

    return y


