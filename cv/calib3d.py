import numpy as np
import numba as nb
from pycontrol.matrix import lie



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



@nb.njit()
def joint_map(colorImgs, depthImgs, poses, K, depthScale):
    fx, fy, cx, cy = K
    pointcloud = []
    for i in range(5):
        color_img = colorImgs[i]
        depth_img = depthImgs[i]
        T = poses[i]                 # Twc
        h, w, channel = color_img.shape

        for v in range(h):
            for u in range(w):
                d = depth_img[v, u]
                if d == 0:
                    continue

                depth = d / depthScale
                x = (u - cx) * depth / fx
                y = (v - cy) * depth / fy
                point = np.array([x, y, depth])
                pointWorld = lie.SE3_mul_p(T, point)   # camera point to world point

                blue = color_img[v, u, 0]
                green = color_img[v, u, 1]
                red = color_img[v, u, 2]

                pointWorld = np.append(pointWorld, [red, green, blue])
                pointcloud.append(pointWorld)

    return pointcloud




