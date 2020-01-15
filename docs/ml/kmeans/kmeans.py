import numpy as np
from pycontrol import ml, params
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



n_samples = 1000
n_features = 2
n_clusters = 4

X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                  centers=[[-1,-1],[0,0],[1,1],[2,2]],
                  cluster_std=[0.4,0.2,0.2,0.2],
                  # centers=[[-1,-1], [-1,1], [1,1], [1,-1]],
                  # cluster_std=[0.2, 0.2, 0.2, 0.2]
                  )

y_pred = ml.kmeans(X, n_clusters=n_clusters, init=params.kmeanspp)


colors = ['red', 'yellow', 'blue', 'green']

plt.figure()
for i in range(n_clusters):
    inds = np.where(y_pred == i)[0]
    plt.scatter(X[inds,0], X[inds,1], color=colors[i])
plt.show()


