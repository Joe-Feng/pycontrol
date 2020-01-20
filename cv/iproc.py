import cv2
import numpy as np
import numba as nb
from pycontrol.math import prob


def padding(image, old_shape, new_shape, mode):
    if len(old_shape) != len(new_shape):
        raise Exception('dim of new_shape should be equal to old_shape!')

    old_h, old_w = old_shape
    new_h, new_w = new_shape

    if new_h <= old_h and new_w <= old_w:
        return image






@nb.njit()
def filterCore(image, shape, ksize, kernel, is_mean=True):
    kh, kw = ksize
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError('ksize should be odd number')
    n = kh*kw

    if len(shape) == 3:
        h, w, channel = shape
    else:
        h, w = shape
        channel = 1

    dst = np.zeros(shape=(h, w, channel), dtype=np.uint8)

    hmin = 0
    hmax = kh - 1
    wmin = 0
    wmax = kw - 1
    ch, cw = int(kh / 2), int(kw / 2)

    for j in range(h - kh + 1):
        for i in range(w - kw + 1):
            domain = image[hmin:hmax+1, wmin:wmax+1]
            s = np.zeros(shape=(channel,))
            for a in range(kh):
                for b in range(kw):
                    s += domain[a,b]*kernel[a,b]

            if is_mean:
                dst[ch, cw, :] = (s / n).astype(np.uint8)
            else:
                dst[ch, cw, :] = s.astype(np.uint8)

            wmin += 1
            wmax += 1
            cw += 1
        wmin = 0
        wmax = kw - 1
        cw = int(kw / 2)

        hmin += 1
        hmax += 1
        ch += 1

    return dst



def blur(image, shape, ksize):
    """
    均值滤波
    --------------------
    Mean filtering
    """
    kernel = np.ones(shape=ksize, dtype=np.uint8)
    dst = filterCore(image, shape, ksize, kernel, is_mean=True)
    return dst


def getGaussianKernel(ksize, sigma1, sigma2):
    kh, kw = ksize
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError('ksize should be odd number')

    hmin = int(kh / 2)
    wmin = int(kw / 2)
    h = np.arange(-hmin, hmin+1)
    w = np.arange(-wmin, wmin+1)
    x, y = np.meshgrid(w, h)

    norm = prob.normal_distribution2d(x, y, sigma1, sigma2)
    norm = norm / np.sum(norm)
    return norm

def GaussianBlur(image, shape, ksize, sigmaX=1.0, sigmaY=1.0):
    """
    高斯滤波
    --------------
    Gaussian filtering
    """
    kernel = getGaussianKernel(ksize, sigmaX, sigmaY)
    dst = filterCore(image, shape, ksize, kernel, is_mean=False)
    return dst

