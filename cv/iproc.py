import numpy as np
import numba as nb
from pycontrol.math import prob
from pycontrol.python import python
from pycontrol import params




def neighbor4(point):
    """
    某点的四邻域上的点
    -------------------------
    The points on the four neighborhoods of a point
    """
    neighbor = []

    left = [point[0], point[1]-1]
    right = [point[0], point[1]+1]
    up = [point[0]-1, point[1]]
    down = [point[0]+1, point[1]]

    neighbor.append(left)
    neighbor.append(right)
    neighbor.append(up)
    neighbor.append(down)

    return np.array(neighbor)



def getKernelSize(ksize):
    if ksize is None:
        return 0,0
    elif isinstance(ksize, int):
        kh = kw = ksize
    else:
        kh, kw = ksize

    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError('ksize should be odd number!')

    return kh, kw



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



@nb.njit()
def cornerNMS(src, ksize, borderType=params.border_default):
    kh, kw = ksize
    shape = src.shape
    dtype = src.dtype
    old_h, old_w = shape[0], shape[1]

    dst = np.zeros(shape=(old_h, old_w), dtype=dtype)

    # padding
    new_h = old_h + kh - 1
    new_w = old_w + kw - 1
    src = padding(src, shape, (new_h, new_w), borderType)
    h, w = new_h, new_w


    hmin = 0
    hmax = kh - 1
    wmin = 0
    wmax = kw - 1
    ch, cw = 0, 0

    for j in range(h - kh + 1):
        for i in range(w - kw + 1):
            domain = src[hmin:hmax+1, wmin:wmax+1]
            maxValue = np.max(domain)
            if src[ch+int(kh/2), cw+int(kw/2)] == maxValue:
                dst[ch, cw] = maxValue

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



@nb.njit()
def padding(image, old_shape, new_shape, borderType=params.border_default, value=None):
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
    dtype = image.dtype

    if new_h <= old_h and new_w <= old_w:
        return image

    # 上下左右各需要填充的行列数
    # Number of rows and columns to be filled
    top_d = int((new_h - old_h) / 2)
    bottom_d = new_h - top_d - old_h
    left_d = int((new_w - old_w) / 2)
    right_d = new_w - left_d - old_w

    # 原图在目标图像中的位置坐标
    # Position coordinates of the original image in the target image
    x0 = left_d
    y0 = top_d
    x1 = x0 + old_w
    y1 = y0 + old_h


    if borderType == params.border_default or \
        borderType == params.border_reflect_101:
        topPadding = python.flipud(image[1:top_d+1])
        image = np.concatenate((topPadding, image), axis=0)
        bottomPadding = python.flipud(image[-1-bottom_d:-1])
        image = np.concatenate((image, bottomPadding), axis=0)
        leftPadding = python.fliplr(image[:, 1:left_d+1])
        image = np.concatenate((leftPadding, image), axis=1)
        rightPadding = python.fliplr(image[:, -1-right_d:-1])
        image = np.concatenate((image, rightPadding), axis=1)

        return image

    elif borderType == params.border_constant:
        if value is None:
            value = 0

        image_mask = (np.ones(shape=mask_shape) * value).astype(dtype)

        image_mask[y0:y1, x0:x1] = image
        return image_mask




@nb.njit()
def filterCore(image, shape, kernel, is_mean=True, borderType=params.border_default):
    kh, kw = kernel.shape
    n = kh*kw

    dtype = image.dtype
    if len(shape) == 3:
        old_h, old_w, channel = shape
        dst = np.zeros(shape=(old_h, old_w, channel), dtype=dtype)
    elif len(shape) == 2:
        old_h, old_w = shape
        channel = 1
        dst = np.zeros(shape=(old_h, old_w), dtype=dtype)

    # padding
    new_h = old_h + kh - 1
    new_w = old_w + kw - 1
    image = padding(image, shape, (new_h, new_w), borderType)
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
                    dst[ch, cw, :] = (s / n).astype(dtype)
                else:
                    dst[ch, cw] = (s / n).astype(dtype)[0]
            else:
                if len(shape) == 3:
                    dst[ch, cw, :] = s.astype(dtype)
                else:
                    dst[ch, cw] = s.astype(dtype)[0]

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



def blur(image, shape, ksize, borderType=params.border_default):
    """
    均值滤波
    --------------------
    Mean filtering
    """
    ksize = getKernelSize(ksize)
    kernel = np.ones(shape=ksize)
    dst = filterCore(image, shape, kernel, is_mean=True, borderType=borderType)
    return dst


def getGaussianKernel(ksize, sigmaX=None, sigmaY=None):
    kh, kw = ksize

    def getKsize(sigma):
        ksize = int(np.round_(6*sigma + 1))
        if ksize % 2 == 0:
            ksize += 1

        return ksize


    if sigmaX is not None and sigmaY is None:
        sigmaY = sigmaX
    elif sigmaX is None and sigmaY is None:
        sigmaX = 0.3 * ((kw - 1) * 0.5 - 1) + 0.8
        sigmaY = 0.3 * ((kh - 1) * 0.5 - 1) + 0.8

    if kh == 0 and kw == 0:
        kh = getKsize(sigmaY)
        kw = getKsize(sigmaX)


    hmin = int(kh / 2)
    wmin = int(kw / 2)
    h = np.arange(-hmin, hmin+1)
    w = np.arange(-wmin, wmin+1)
    x, y = np.meshgrid(w, h)

    norm = prob.normal_distribution2d(x, y, sigmaX, sigmaY)
    norm = norm / np.sum(norm)
    return norm


def GaussianBlur(image, shape, ksize, sigmaX=None, sigmaY=None, borderType=params.border_default):
    """
    高斯滤波
    --------------
    Gaussian filtering
    """
    ksize = getKernelSize(ksize)
    kernel = getGaussianKernel(ksize, sigmaX, sigmaY)
    dst = filterCore(image, shape, kernel, is_mean=False, borderType=borderType)
    return dst


@nb.njit()
def pyrDown(image):
    """
    下采样，删除偶数行和列
    ---------------------------
    Downsampling, deleting even rows and columns
    """
    dst = image[::2,::2]
    return dst

