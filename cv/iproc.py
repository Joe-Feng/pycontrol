import numpy as np
import numba as nb
from pycontrol.math import prob
from pycontrol import params



@nb.njit()
def padding(image, old_shape, new_shape, borderType=params.border_constant, value=None):
    """
    增大图片并填充边缘
    -----------------------
    Increase the image and fill the edge
    """
    if len(old_shape) == 2:
        old_h, old_w = old_shape
        channel = 1
        mask_shape = new_shape
    else:
        old_h, old_w, channel = old_shape
        mask_shape = new_shape + (channel,)
    new_h, new_w = new_shape

    if new_h <= old_h and new_w <= old_w:
        return image

    dx = int((new_w - old_w) / 2)
    dy = int((new_h - old_h) / 2)

    x0 = dx
    y0 = dy
    x1 = x0 + old_w
    y1 = y0 + old_h


    if borderType == params.border_constant:
        image_mask = (np.ones(shape=mask_shape) * value).astype(np.uint8)

        image_mask[y0:y1, x0:x1] = image
        return image_mask




@nb.njit()
def filterCore(image, shape, kernel, is_mean=True, borderType=params.border_constant):
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError('ksize should be odd number!')
    n = kh*kw

    if len(shape) == 3:
        old_h, old_w, channel = shape
        dst = np.zeros(shape=(old_h, old_w, channel), dtype=np.uint8)
    else:
        old_h, old_w = shape
        channel = 1
        dst = np.zeros(shape=(old_h, old_w), dtype=np.uint8)

    # padding
    new_h = old_h + kh - 1
    new_w = old_w + kw - 1
    image = padding(image, shape, (new_h, new_w), borderType, value=0)
    h, w = new_h, new_w


    hmin = 0
    hmax = kh - 1
    wmin = 0
    wmax = kw - 1
    ch, cw = 0, 0

    for j in range(h - kh + 1):
        for i in range(w - kw + 1):
            domain = image[hmin:hmax+1, wmin:wmax+1]
            s = np.zeros(shape=(channel,))
            for a in range(kh):
                for b in range(kw):
                    s += domain[a,b]*kernel[a,b]

            if is_mean:
                if len(shape) == 3:
                    dst[ch, cw, :] = (s / n).astype(np.uint8)
                else:
                    dst[ch, cw] = (s / n).astype(np.uint8)[0]
            else:
                if len(shape) == 3:
                    dst[ch, cw, :] = s.astype(np.uint8)
                else:
                    dst[ch, cw] = s.astype(np.uint8)[0]

            wmin += 1
            wmax += 1
            cw += 1
        wmin = 0
        wmax = kw - 1
        cw = 0

        hmin += 1
        hmax += 1
        ch += 1

    return dst



def blur(image, shape, ksize, borderType=params.border_constant):
    """
    均值滤波
    --------------------
    Mean filtering
    """
    kernel = np.ones(shape=ksize, dtype=np.uint8)
    dst = filterCore(image, shape, kernel, is_mean=True, borderType=borderType)
    return dst


def getGaussianKernel(ksize, sigma1, sigma2):
    kh, kw = ksize
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError('ksize should be odd number!')

    hmin = int(kh / 2)
    wmin = int(kw / 2)
    h = np.arange(-hmin, hmin+1)
    w = np.arange(-wmin, wmin+1)
    x, y = np.meshgrid(w, h)

    norm = prob.normal_distribution2d(x, y, sigma1, sigma2)
    norm = norm / np.sum(norm)
    return norm


def GaussianBlur(image, shape, ksize, sigmaX=1.0, sigmaY=1.0, borderType=params.border_constant):
    """
    高斯滤波
    --------------
    Gaussian filtering
    """
    kernel = getGaussianKernel(ksize, sigmaX, sigmaY)
    dst = filterCore(image, shape, kernel, is_mean=False, borderType=borderType)
    return dst


def getSobel(ksize):
    Gx = np.array(
        [[-1,0,1],
         [-2,0,2],
         [-1,0,1]]
    )
    Gy = np.array(
        [[-1,-2,-1],
         [0,0,0],
         [1,2,1]]
    )

    return np.array([Gx, Gy])


# @nb.njit()
def cornerEigenValsAndVecs(image, shape, kernel, k, borderType=params.border_constant):
    _, kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError('ksize should be odd number!')

    Gx = kernel[0]
    Gy = kernel[1]

    old_h, old_w = shape
    dst = np.zeros(shape=(old_h, old_w))

    # padding
    new_h = old_h + kh - 1
    new_w = old_w + kw - 1
    image = padding(image, shape, (new_h, new_w), borderType, value=0)
    h, w = new_h, new_w

    hmin = 0
    hmax = kh - 1
    wmin = 0
    wmax = kw - 1
    ch, cw = 0, 0

    for j in range(h - kh + 1):
        for i in range(w - kw + 1):
            domain = image[hmin:hmax + 1, wmin:wmax + 1]
            dx = 0
            dy = 0
            for a in range(kh):
                for b in range(kw):
                    dx += domain[a, b] * Gx[a, b]
                    dy += domain[a, b] * Gy[a, b]

            M = np.array([[dx*dx, dx*dy],
                          [dx*dy, dy*dy]])
            lambda_1, lambda_2 = np.linalg.eigvals(M)
            det = lambda_1 * lambda_2
            trace = lambda_1 + lambda_2
            R = det - k*trace*trace
            dst[ch, cw] = R

            wmin += 1
            wmax += 1
            cw += 1
        wmin = 0
        wmax = kw - 1
        cw = 0

        hmin += 1
        hmax += 1
        ch += 1

    return dst


def cornerHarris(image, blockSize, ksize, k, borderType=params.border_constant):
    """
    harris角点检测
    ----------------------
    harris corner detect
    """
    if len(image.shape) == 3:
        raise Exception('need grayscale image!')

    sobel = getSobel(ksize)
    dst = cornerEigenValsAndVecs(image, image.shape, sobel, k, borderType)

    return dst