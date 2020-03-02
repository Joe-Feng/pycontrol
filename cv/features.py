import numpy as np
import numba as nb
import cv2
from pycontrol import params
from pycontrol.cv import iproc




@nb.njit()
def harrisGradient(image, kernel, borderType=params.border_default):
    _, kh, kw = kernel.shape

    Gx = kernel[0]
    Gy = kernel[1]

    shape = image.shape
    old_h, old_w = shape
    Ix = np.zeros(shape=(old_h, old_w))
    Iy = np.zeros(shape=(old_h, old_w))

    # padding
    new_h = old_h + kh - 1
    new_w = old_w + kw - 1
    image = iproc.padding(image, shape, (new_h, new_w), borderType)
    h, w = new_h, new_w


    hmin = 0
    hmax = kh - 1
    wmin = 0
    wmax = kw - 1
    ch, cw = 0, 0

    for j in range(h - kh + 1):
        for i in range(w - kw + 1):
            domain = image[hmin:hmax + 1, wmin:wmax + 1]

            # dx = 0
            # dy = 0
            # for a in range(kh):
            #     for b in range(kw):
            #         dx += domain[a, b] * Gx[a, b]
            #         dy += domain[a, b] * Gy[a, b]

            dx = np.sum(domain*Gx)
            dy = np.sum(domain*Gy)

            Ix[ch, cw] = dx
            Iy[ch, cw] = dy

            wmin += 1
            wmax += 1
            cw += 1

        wmin = 0
        wmax = kw - 1
        cw = 0

        hmin += 1
        hmax += 1
        ch += 1

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    return Ix2, Iy2, Ixy



def cornerHarris(image, ksize, k, borderType=params.border_default):
    """
    harris角点检测
    ----------------------
    harris corner detect
    """
    if len(image.shape) != 2:
        raise Exception('need grayscale image!')

    sobel = iproc.getSobel(ksize)

    # 计算梯度
    # Calculate the gradient
    Ix2, Iy2, Ixy = harrisGradient(image, sobel, borderType)

    # 高斯加权
    # Gaussian weighting
    Ix2 = iproc.GaussianBlur(Ix2, Ix2.shape, ksize=(ksize, ksize))
    Iy2 = iproc.GaussianBlur(Iy2, Iy2.shape, ksize=(ksize, ksize))
    Ixy = iproc.GaussianBlur(Ixy, Ixy.shape, ksize=(ksize, ksize))

    # 计算角点响应值R
    # Calculate corner response value R
    detM = Ix2 * Iy2 - Ixy * Ixy
    traceM = Ix2 + Iy2
    R = detM - k * traceM * traceM

    return R



def gaussDiffPyramid(image, O, S):
    """
    构建高斯差分金字塔
    -------------------
    Constructing a Gaussian Difference Pyramid
    """
    # 计算每层的sigma
    # Calculate the sigma value of each layer
    m = S - 3
    sigma0 = 1.52
    k = 2
    layerSigmas = []
    for layer in range(S):
        if layer == 0:
            layerSigmas.append(sigma0)
        else:
            preSigma = sigma0 * np.power(k, (layer - 1) / m)
            curSigma = sigma0 * np.power(k, layer / m)
            diffSigma = np.sqrt(curSigma * curSigma - preSigma * preSigma)
            layerSigmas.append(diffSigma)


    # 生成高斯金字塔
    # Generate Gaussian Pyramid
    pyrImages = []
    for octave in range(O):
        layerImages = []
        for layer in range(S):
            # 对于第一组第一层图片，
            # 由原图直接滤波得到
            # For the first set of first layer pictures，
            # Directly from the original image filtering
            if octave == 0 and layer == 0:
                sigma = layerSigmas[0]

                curImage = iproc.GaussianBlur(
                    image, image.shape, ksize=None,
                    sigmaX=sigma, sigmaY=sigma
                )
                layerImages.append(curImage)

            # 第一层的图片由上一组的倒数第三张下采样得到，
            # 不需要滤波
            # The first layer of pictures is obtained by downsampling from
            # the third to last picture of the previous group,
            # no filtering is required
            elif layer == 0:
                curImage = pyrImages[octave-1][-3]
                curImage = iproc.pyrDown(curImage)
                layerImages.append(curImage)

            # 对于非第一层图片，用相对尺度对上一层图片滤波得到
            # For non-first-layer pictures, use the relative scale to filter the previous picture
            else:
                diffSigma = layerSigmas[layer]

                preImage = layerImages[-1]
                curImage = iproc.GaussianBlur(
                    preImage, preImage.shape, ksize=None,
                    sigmaX=diffSigma, sigmaY=diffSigma
                )
                layerImages.append(curImage)

        pyrImages.append(layerImages)


    # 高斯差分金字塔
    # Gaussian Difference Pyramid
    DOGPyrImages = nb.typed.List()
    for octave in range(O):
        DOGLayerImages = []
        for layer in range(1,S):
            diffImage = pyrImages[octave][layer] - pyrImages[octave][layer-1]
            DOGLayerImages.append(diffImage)

        DOGLayerImages = np.array(DOGLayerImages)
        DOGPyrImages.append(DOGLayerImages)

    return DOGPyrImages


@nb.njit()
def isSiftExtremePoint(DOGPyrImages, octave, layer, j, i):
    """
    判断某个像素是否是3*3*3邻域内的极值
    ------------------------------------
    Determine if a pixel is an extreme value in a 3*3*3 neighborhood
    """
    domain = np.zeros(shape=(3,3,3))
    domain[:,:,0] = DOGPyrImages[octave][layer - 1][j - 1:j + 2, i - 1:i + 2]
    domain[:,:,1] = DOGPyrImages[octave][layer][j - 1:j + 2, i - 1:i + 2]
    domain[:,:,2] = DOGPyrImages[octave][layer + 1][j - 1:j + 2, i - 1:i + 2]

    maxValue = np.max(domain)
    minValue = np.min(domain)
    value = DOGPyrImages[octave][layer][j,i]

    if value == maxValue or value == minValue:
        return True
    return False



@nb.njit()
def siftKeyPoints(DOGPyrImages, O, S):
    """
    在高斯差分金字塔中寻找极值点
    ------------------------------
    Find extreme points in a Gaussian difference pyramid
    """
    m = S - 3
    T = 0.04
    threshold = 255*0.5*T/m
    SIFT_IMG_BORDER = 5

    keyPoints = []
    for octave in range(O):
        shape = DOGPyrImages[octave][0].shape
        h, w = shape
        for layer in range(1, S-2):
            for j in range(SIFT_IMG_BORDER, h-SIFT_IMG_BORDER):
                for i in range(SIFT_IMG_BORDER, w-SIFT_IMG_BORDER):
                    value = DOGPyrImages[octave][layer][j,i]
                    if np.abs(value) > threshold:
                        if isSiftExtremePoint(DOGPyrImages, octave, layer, j, i):
                            point = siftInterpExtreme(
                                DOGPyrImages, octave, layer, j, i,
                                S, SIFT_IMG_BORDER, T)
                            if point is not None:
                                keyPoints.append(point)

    return keyPoints


@nb.njit()
def siftJacobi(DOGPyrImages, octave, layer, j, i):
    J = np.ones(shape=(3,1))

    J[0][0] = \
        (DOGPyrImages[octave][layer][j,i+1]
         - DOGPyrImages[octave][layer][j,i-1]) / 2

    J[1][0] = \
        (DOGPyrImages[octave][layer][j+1, i]
         - DOGPyrImages[octave][layer][j-1, i]) / 2

    J[2][0] = \
        (DOGPyrImages[octave][layer+1][j, i]
         - DOGPyrImages[octave][layer-1][j, i]) / 2

    return J


@nb.njit()
def siftHessian(DOGPyrImages, octave, layer, j, i):
    H = np.zeros(shape=(3,3))

    d2f_dx2 = \
        DOGPyrImages[octave][layer][j,i+1] \
         + DOGPyrImages[octave][layer][j,i-1] \
         - 2*DOGPyrImages[octave][layer][j,i]

    d2f_dy2 = \
        DOGPyrImages[octave][layer][j+1, i] \
         + DOGPyrImages[octave][layer][j-1, i] \
         - 2 * DOGPyrImages[octave][layer][j, i]

    d2f_dsigma2 = \
        DOGPyrImages[octave][layer+1][j, i] \
        + DOGPyrImages[octave][layer-1][j, i] \
        - 2 * DOGPyrImages[octave][layer][j, i]

    d2f_dxdy = \
        (DOGPyrImages[octave][layer][j+1, i+1]
         + DOGPyrImages[octave][layer][j-1, i-1]
         - DOGPyrImages[octave][layer][j+1, i-1]
         - DOGPyrImages[octave][layer][j-1, i+1]) / 4

    d2f_dxdsigma = \
        (DOGPyrImages[octave][layer+1][j, i+1]
         + DOGPyrImages[octave][layer-1][j, i-1]
         - DOGPyrImages[octave][layer+1][j, i-1]
         - DOGPyrImages[octave][layer-1][j, i+1]) / 4

    d2f_dydsigma = \
        (DOGPyrImages[octave][layer + 1][j+1, i]
         + DOGPyrImages[octave][layer - 1][j-1, i]
         - DOGPyrImages[octave][layer + 1][j-1, i]
         - DOGPyrImages[octave][layer - 1][j+1, i]) / 4

    H[0][0] = d2f_dx2
    H[0][1] = H[1][0] = d2f_dxdy
    H[0][2] = H[2][0] = d2f_dxdsigma
    H[1][1] = d2f_dy2
    H[1][2] = H[2][1] = d2f_dydsigma
    H[2][2] = d2f_dsigma2

    return H



@nb.njit()
def siftEdgeEffect(H):
    """
    去除边缘效应
    ---------------
    Remove edge effects
    """
    gamma = 10.0
    traceH = H[0][0] + H[1][1]
    detH = H[0][0]*H[1][1] - H[0][1]*H[1][0]

    if detH < 0:
        return True

    if np.power(traceH, 2) / detH >= np.power(gamma+1, 2) / gamma:
        return True

    return False


@nb.njit()
def siftInterpExtreme(DOGPyrImages, octave, layer, j, i,
                      S, SIFT_IMG_BORDER, T):
    """
    用插值法求极值点
    ------------------
    Finding extreme points by interpolation
    """
    SIFT_MAX_INTERP_STEPS = 5
    m = S - 3
    threshold = 255*T / m
    old_value = DOGPyrImages[octave][layer][j, i]
    shape = DOGPyrImages[octave][0].shape

    step = 0
    for _ in range(SIFT_MAX_INTERP_STEPS):
        df = siftJacobi(DOGPyrImages, octave, layer, j, i)
        H = siftHessian(DOGPyrImages, octave, layer, j, i)
        H_inv = np.linalg.inv(H)
        off_X = -1 * np.dot(H_inv, df)

        off_x, off_y, off_sigma = off_X[0,0], off_X[1,0], off_X[2,0]
        if np.abs(off_x)<0.5 and np.abs(off_y)<0.5 and np.abs(off_sigma)<0.5:
            break

        i = i + int(np.round_(off_x))
        j = j + int(np.round_(off_y))
        layer = layer + int(np.round_(off_sigma))

        # 超出边界
        # Out of bounds
        if layer<1 or layer>m \
            or j < SIFT_IMG_BORDER or i < SIFT_IMG_BORDER \
            or j >= shape[0]-SIFT_IMG_BORDER \
            or i >= shape[1]-SIFT_IMG_BORDER:
            return None

        step += 1

    # 超出最大循环次数
    # Exceeded the maximum number of cycles
    if step == SIFT_MAX_INTERP_STEPS:
        return None


    # 计算新的极值
    # Calculate new extreme values
    df_T = np.transpose(df)
    new_value = old_value + 0.5*np.dot(df_T, off_X)[0][0]

    # 去除低对比度的点
    # Remove low-contrast points
    if np.abs(new_value) < threshold:
        return None

    # 去除边缘效应
    # Remove edge effects
    if siftEdgeEffect(H):
        return None

    # 计算新的极值点
    # Calculating new extreme points
    new_i = i + off_x
    new_j = j + off_y
    new_layer = layer + off_sigma

    # 将关键点坐标还原到原图像大小的尺度上
    # Restore the key point coordinates to the scale of the original image size
    scale = np.power(2, octave)
    new_i *= scale
    new_j *= scale

    return np.array([new_j, new_i])



def siftDetectAndCompute(image):
    """
    SIFT特征点检测
    -------------------
    SIFT feature point detection
    """
    shape = image.shape
    if len(shape) != 2:
        raise Exception('need grayscale image!')

    h, w = shape
    image = image.astype(np.float)

    # 计算组数
    # Calculate the number of groups
    O = int(np.log2(min(h, w)) - 3)
    # 计算每组的层数
    # Calculate the number of layers in each group
    m = 3
    S = m + 3

    # 构建高斯差分金字塔
    # Constructing a Gaussian Difference Pyramid
    DOGPyrImages = gaussDiffPyramid(image, O=O, S=S)
    keyPoints = siftKeyPoints(DOGPyrImages, O=O, S=S)
    return keyPoints




def orbPyrImages(image, scaleFactor, nlevels):
    """
    生成图像金字塔
    --------------------
    Generate Image Pyramid
    """
    h, w = image.shape
    pyrImages = nb.typed.List()
    pyrImages.append(image)
    for n in range(1, nlevels):
        scale = np.power(scaleFactor, n)
        new_h, new_w = int(h/scale), int(w/scale)
        image_resized = cv2.resize(
            image, (new_w,new_h), interpolation=cv2.INTER_LINEAR)
        pyrImages.append(image_resized)

    return pyrImages


@nb.njit()
def orbPreKeyPoints(image, h, w, fastThreshold):
    """
    根据上下左右四个点选取预角点
    -----------------------------
    Select the pre-angle point according to the four points
    """
    p = image[h, w]
    p1 = image[h - 3, w]
    p5 = image[h, w + 3]
    p9 = image[h + 3, w]
    p13 = image[h, w - 3]

    surround = np.array([p1, p5, p9, p13])

    # 判断是否为角点
    # Determine if it is a corner point
    if np.sum(surround > p + fastThreshold) >= 3 or \
        np.sum(surround < p - fastThreshold) >= 3:
        return True
    else:
        return False


@nb.njit()
def orbSurround(image, h, w):
    """
    返回点p周围的16个点
    ------------------------
    Returns 16 points around point p
    """
    p1 = image[h-3, w]
    p2 = image[h-3, w+1]
    p3 = image[h-2, w+2]
    p4 = image[h-1, w+3]
    p5 = image[h, w+3]
    p6 = image[h+1, w+3]
    p7 = image[h+2, w+2]
    p8 = image[h+3, w+1]
    p9 = image[h+3, w]
    p10 = image[h+3, w-1]
    p11 = image[h+2, w-2]
    p12 = image[h+1, w-3]
    p13 = image[h, w-3]
    p14 = image[h-1, w-3]
    p15 = image[h-2, w-2]
    p16 = image[h-3, w-1]

    surround = np.array(
        [p1,p2,p3,p4,p5,p6,p7,p8,
         p9,p10,p11,p12,p13,p14,p15,p16]
    )

    return surround

@nb.njit()
def isOrbKeyPoints(p, surround, fastThreshold):
    """
    判断点p周围是否有连续n个点大于阈值
    -------------------------------------
    Determine if there are n consecutive points around point p greater than the threshold
    """
    num = 0
    moreThan = (surround > p + fastThreshold)
    for boolean in moreThan:
        if boolean:
            num += 1
            if num == 9:
                return True
        else:
            num = 0


    num = 0
    lessThan = (surround < p - fastThreshold)
    for boolean in lessThan:
        if boolean:
            num += 1
            if num == 9:
                return True
        else:
            num = 0

    return False


@nb.njit()
def orbKeyPoints(pyrImages, scaleFactor, fastThreshold):
    """
    使用FAST算法检测特征点
    -------------------------
    Detect feature points using FAST algorithm
    """
    src_h, src_w = pyrImages[0].shape
    dst = np.zeros(shape=(src_h, src_w))

    for n, img in enumerate(pyrImages):
        scale = np.power(scaleFactor, n)
        old_h, old_w = img.shape
        new_h, new_w = old_h+6, old_w+6
        image = iproc.padding(img, (old_h,old_w), (new_h,new_w))
        h, w = new_h, new_w
        ch, cw = 0, 0

        for j in range(h-6):
            for i in range(w-6):
                pad_ch = ch + 3
                pad_cw = cw + 3

                if orbPreKeyPoints(image, pad_ch, pad_cw, fastThreshold):
                    surround = orbSurround(image, pad_ch, pad_cw)
                    # 判断是否为角点
                    # Determine if it is a corner point
                    p = image[pad_ch, pad_cw]
                    if isOrbKeyPoints(p, surround, fastThreshold):
                        V = np.sum(np.abs(surround - p))

                        scaled_h = min(int(ch * scale), src_h-1)
                        scaled_w = min(int(cw * scale), src_w-1)
                        dst[scaled_h, scaled_w] = V

                cw += 1

            cw = 0
            ch += 1

    # 对特征点做非极大值抑制
    # Non-maximum suppression of feature points
    dst_nms = iproc.cornerNMS(dst, ksize=(7,7))

    return dst_nms



def orbDetectAndCompute(image, fastThreshold=20):
    """
    ORB特征点检测
    -----------------
    ORB feature point detection
    """
    shape = image.shape
    if len(shape) != 2:
        raise Exception('need grayscale image!')

    # 金字塔尺度
    # Pyramid scale
    scaleFactor = 1.2

    # 金字塔层数
    # Number of pyramid layers
    nlevels = 8

    # 生成图像金字塔
    # Generate Image Pyramid
    pyrImages = orbPyrImages(image, scaleFactor, nlevels)

    # 使用FAST算法检测特征点
    # Detect feature points using FAST algorithm
    keyPoints = orbKeyPoints(pyrImages, scaleFactor, fastThreshold)

    return keyPoints
