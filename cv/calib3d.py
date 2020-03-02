import numpy as np
import numba as nb
from pycontrol.matrix import lie



def camera2pixel(K, Pc):
    '''
    相机坐标系转换为像素坐标系
    K: [fx,fy,cx,cy]
    Pc：对Z归一化的相机坐标系坐标
    返回：像素坐标系坐标
    -----------------------------------
    Camera coordinate system to pixel coordinate system
    Pc: Camera coordinate system coordinates normalized to Z
    return: Pixel coordinate system coordinates
    '''
    fx, fy, cx, cy = K
    x, y = Pc[:2]

    u = fx * x + cx
    v = fy * y + cy
    return np.array([u, v])


def pixel2camera(K, Puv):
    """
    像素坐标系转换为相机坐标系
    K: [fx,fy,cx,cy]
    --------------------------------
    Convert pixel coordinate system to camera coordinate system
    """
    fx, fy, cx, cy = K
    u = Puv[0]
    v = Puv[1]

    X = (u - cx) / fx
    Y = (v - cy) / fy

    return np.array([X, Y, 1])


@nb.njit()
def undistort(image, K, distort):
    """
    图像去畸变
    K: [fx,fy,cx,cy]
    distort: 畸变参数, [k1,k2,p1,p2]
    -------------------------------
    undistort image
    distort: Distortion parameter
    """
    fx, fy, cx, cy = K
    k1, k2, p1, p2 = distort
    h, w = image.shape[:2]
    image_undistorted = np.zeros(shape=(h, w), dtype=np.uint8)

    for v in range(h):
        for u in range(w):
            x = (u - cx) / fx
            y = (v - cy) / fy
            r = np.sqrt(x ** 2 + y ** 2)
            x_distorted = x * (1 + k1 * (r ** 2) + k2 * (r ** 4)) + 2 * p1 * x * y + p2 * (r ** 2 + 2 * (x ** 2))
            y_distorted = y * (1 + k1 * (r ** 2) + k2 * (r ** 4)) + 2 * p2 * x * y + p1 * (r ** 2 + 2 * (y ** 2))

            u_undistorted = fx * x_distorted + cx
            v_undistorted = fy * y_distorted + cy

            if 0 < u_undistorted < w and 0 < v_undistorted < h:
                image_undistorted[v, u] = image[int(v_undistorted), int(u_undistorted)]
            else:
                image_undistorted[v, u] = 0

    return image_undistorted


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




