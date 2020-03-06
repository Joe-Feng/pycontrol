import numpy as np
from pycontrol import params



def normalize(X, alpha=None, beta=None, axis=None, normType=params.norm_L2):
    '''
    规范化
    '''
    if axis is not None:
        keepdims = True
    else:
        keepdims = False

    if normType == params.norm_L2:
        L2 = np.sqrt(np.sum(np.power(X, 2), axis=axis, keepdims=keepdims))
        norm = X / L2
        return norm

    if normType == params.norm_minmax:
        minValue = np.min(X, axis=axis, keepdims=keepdims)
        maxValue = np.max(X, axis=axis, keepdims=keepdims)

        if alpha is None and beta is None:
            norm = (X - minValue) / (maxValue - minValue)
        else:
            norm = (X - minValue) * (beta - alpha) / (maxValue - minValue) + alpha
        return norm

