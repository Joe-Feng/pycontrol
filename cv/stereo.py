import numpy as np
import numba as nb



def SGBM():
    #TODO 完成
    pass


@nb.njit()
def computeDepth(disparity, left_image, K, b):
    """
    根据视差计算深度
    ---------------------
    Calculate depth from parallax
    """
    fx, fy, cx, cy = K
    h, w = disparity.shape[:2]
    pointcloud = []
    for v in range(h):
        for u in range(w):
            if disparity[v, u] <= 0 or disparity[v, u] >= 96.0:
                continue

            x = (u-cx)/fx
            y = (v-cy)/fy
            depth = fx*b/disparity[v, u]
            x *= depth
            y *= depth

            # x, y, depth, color
            point = [x, y, depth, left_image[v,u]/255.0]
            pointcloud.append(point)

    return pointcloud


