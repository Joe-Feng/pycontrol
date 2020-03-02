import numpy as np
from pycontrol import params



def normalize(X, alpha=None, beta=None, normType=params.norm_L2, dataType=params.batch_feature):
    '''
    规范化
    '''
    if X.ndim == 1:
        X = X[np.newaxis, ...]

    if normType == params.norm_L2:
        L2 = np.sqrt(np.sum(np.power(X, 2), axis=1, keepdims=True))
        norm = X / L2
        return norm

    if normType == params.norm_minmax:
        if dataType == params.batch_feature:
            minValue = np.min(X, axis=1, keepdims=True)
            maxValue = np.max(X, axis=1, keepdims=True)

            norm = (X - minValue)*(beta - alpha) / (maxValue - minValue) + alpha
            return norm

        elif dataType == params.image:
            minValue = np.min(X)
            maxValue = np.max(X)

            norm = (X - minValue)*(beta - alpha) / (maxValue - minValue) + alpha
            return norm



