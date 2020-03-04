import numpy as np
import numba as nb




@nb.njit()
def flipud(src):
    """
    上下翻转
    """
    shape = src.shape
    dtype = src.dtype

    h = shape[0]
    dst = np.zeros(shape=shape, dtype=dtype)

    for i in range(h):
        dst[h-i-1: h-i] = src[i: i+1]

    return dst


@nb.njit()
def fliplr(src):
    """
    左右翻转
    """
    shape = src.shape
    dtype = src.dtype

    w = shape[1]
    dst = np.zeros(shape=shape, dtype=dtype)

    for i in range(w):
        dst[:, w-i-1: w-i] = src[:, i:i + 1]

    return dst

