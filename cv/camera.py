import numpy as np
from pycontrol.matrix import transform



def camera2pixel(K, Pc):
    '''
    相机坐标系转换为像素坐标系
    K: [fx  0  cx
        0  fy  cy
        0  0   1]
    Pc：对Z归一化的相机坐标系坐标
    返回：像素坐标系坐标
    -----------------------------------
    Camera coordinate system to pixel coordinate system
    Pc: Camera coordinate system coordinates normalized to Z
    return: Pixel coordinate system coordinates
    '''
    p = transform.rotate_R(K, Pc)
    return p



