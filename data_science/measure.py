import numpy as np



def euclidean_dist(array1, array2=None):
    """
    计算欧几里得距离
    array1, array2: 必须是2维数组 [batch, features]
    ---------------------
    Calculate Euclid distance
    array1, array2: Must be a 2D array as [batch, features]
    """
    if array2 is None:
        array2 = array1.copy()

    if np.ndim(array1) != 2 or np.ndim(array2) != 2:
        raise ValueError('array1 and array2 must be 2 dims')

    square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
    square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
    square_dist = square1 + square2 - 2*np.matmul(array1, array2.T)
    square_dist[square_dist < 0] = 0
    dist = np.sqrt(square_dist)

    return dist


def SSE(array1, array2):
    """
    误差平方和
    array1, array2: 必须是2维数组 [batch, features]
    -------------
    Sum of Squared Error
    array1, array2: Must be a 2D array as [batch, features]
    """
    if array2 is None:
        array2 = array1.copy()

    if np.ndim(array1) != 2 or np.ndim(array2) != 2:
        raise ValueError('array1 and array2 must be 2 dims')

    square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
    square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
    square_dist = square1 + square2 - 2 * np.matmul(array1, array2.T)
    square_dist[square_dist < 0] = 0

    return square_dist


def cluster_center(X):
    """
    求一簇数据的中心
    -------------------
    Find the center of a cluster of data
    """
    return np.mean(X, axis=0)




