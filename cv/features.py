import numpy as np




def cornerHarris(image):
    """
    harris角点检测
    ---------------------
    Harris corner detection
    """
    shape = image.shape
    if len(shape) != 2:
        raise Exception('need grayscale image!')

    h, w = shape

    dst = np.zeros(shape=(h, w))
