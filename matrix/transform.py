import numpy as np
from pycontrol.matrix import judge
from pycontrol.matrix import matrix as mat
from pycontrol import params
from pycontrol.math import complex
from pycontrol.data_science import dproc
import numba as nb


def rotate_R(R, p):
    '''
    旋转变换
    ------------
    Rotation transformation
    '''
    # TODO 判断是否为正交阵
    # if not judge.is_orthogona_matrix(R):
    #     raise Exception("rotate matrix must be orthogona matrix")

    return np.matmul(R, p)


def rotate_axis(theta, axis, p=None):
    '''
    绕轴axis旋转theta角
    -----------------------
    Rotate theta angle around axis
    '''
    if axis.upper() == params.axis_X:
        R = np.array([[1, 0, 0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta), np.cos(theta)]])
    elif axis.upper() == params.axis_Y:
        R = np.array([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])
    elif axis.upper() == params.axis_Z:
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
    else:
        raise Exception('axis must be in [%s, %s, %s]'
                        % params.axis_X, params.axis_Y, params.axis_Z)

    if not p:
        return R

    return rotate_R(R, p)



def transform_Rt(R, t, p):
    '''
    变换矩阵
    ----------
    Transformation matrix
    '''
    p_r = rotate_R(R, p)
    p_t = p_r + t
    return p_t


def transform_homogeneous(T, p):
    '''
    齐次变换矩阵
    ----------------
    Homogeneous transformation matrix
    '''
    if not judge.is_ndarray(p):
        p = np.array(p)

    if p.shape == (3,):
        p = np.concatenate((p, [1]), axis=0)
    elif p.shape == (3, 1):
        p = np.concatenate((p, [[1]]), axis=0)

    return np.matmul(T, p)


def transform_axis(theta, axis, t, p):
    '''
    绕轴axis旋转theta角，平移向量t
    ---------------------------------
    Rotate theta angle around axis, translate vector t
    '''
    p_r = rotate_axis(theta, axis, p)
    p_t = p_r + t
    return p_t


def inverse_transform(T):
    '''
    变换矩阵的逆
    -----------------
    Inverse of transformation matrix
    '''
    R, t = T2Rt(T)
    R_T = R.transpose()
    T_inverse = Rt2T(R_T, np.matmul(-R_T, t))
    return T_inverse


def Rt2T(R, t):
    '''
    将R，t组合为齐次变换矩阵
    ------------------------
    Combine R, t into homogeneous transformation matrix
    '''
    if not judge.is_ndarray(t):
        t = np.array(t)

    if t.shape == (3,):
        t = t[np.newaxis, :]
        t = t.transpose()
    elif t.shape == (3,1):
        t = t.transpose()

    conc = np.concatenate((R, t), axis=1)
    T = np.concatenate((conc, [[0,0,0,1]]), axis=0)
    return T


def T2Rt(T):
    '''
    将齐次变换矩阵分解为R，t
    ---------------------------
    Decomposition of homogeneous transformation matrix into R, t
    '''
    R = T[0:3, 0:3]
    t = T[0:3, 3:4]
    return R, t


def rotate_xyz_fixed(theta_x, theta_y, theta_z, p=None):
    '''
    X-Y-Z固定角坐标系。绕X,Y,Z轴旋转
    -----------------------------------
    X-Y-Z fixed angle coordinate system. Rotate around x, y, Z axis
    '''
    R_zy = np.matmul(rotate_axis(theta_z, params.axis_Z),
                     rotate_axis(theta_y, params.axis_Y))
    R_zyx = np.matmul(R_zy,
                      rotate_axis(theta_x, params.axis_X))

    if not p:
        return R_zyx

    return rotate_R(R_zyx, p)


def inverse_rotate_xyz(R_zyx):
    '''
    求X-Y-Z旋转的逆解
    已知旋转矩阵，求出分别绕X,Y,Z轴旋转的角度
    -------------------------------------------
    Inverse solution of X-Y-Z rotation
    Given the rotation matrix, find out the angles of rotation around the X, y and Z axes respectively
    '''
    theta_y = np.arctan2(-R_zyx[2,0], np.sqrt(np.power(R_zyx[0,0],2) + np.power(R_zyx[1,0],2)))
    cos_theta_y = np.cos(theta_y)

    if cos_theta_y != 0:
        theta_z = np.arctan2(R_zyx[1,0]/cos_theta_y, R_zyx[0,0]/cos_theta_y)
        theta_x = np.arctan2(R_zyx[2,1]/cos_theta_y, R_zyx[2,2]/cos_theta_y)

    else:
        result = []

        theta_y = np.radians(90)
        theta_z = np.radians(0)
        theta_x = np.arctan2(R_zyx[0,1], R_zyx[1,1])
        result.append([theta_x, theta_y, theta_z])

        theta_y = np.radians(-90)
        theta_z = np.radians(0)
        theta_x = -np.arctan2(R_zyx[0, 1], R_zyx[1, 1])
        result.append([theta_x, theta_y, theta_z])

        return result

    return theta_x, theta_y, theta_z



def rotate_zyx_dynamic(theta_z, theta_y, theta_x, p=None):
    '''
    Z-Y-X 欧拉角，绕Z,Y,X轴旋转
    ------------------------------
    Z-y-x Euler angle, rotating around Z, y, X axis
    '''
    return rotate_xyz_fixed(theta_x, theta_y, theta_z, p)



def inverse_rotate_zyx(R_xyz):
    '''
    求Z-Y-X旋转的逆解
    已知旋转矩阵，求出分别绕Z,Y,X轴旋转的角度
    -------------------------------------------
    Inverse solution of z-y-x rotation
    Given the rotation matrix, find out the rotation angles around the Z, y, and X axes respectively
    '''
    return inverse_rotate_xyz(R_xyz)[::-1]



def rotate_axis_angle(axis_vector, theta, p=None):
    '''
    轴角转换为旋转矩阵（罗格里格斯公式）
    -------------------------
    The transformation of axis angle into rotation matrix (Rodrigues's formula)
    '''
    axis_vector = dproc.normalize(axis_vector, normType=params.norm_L2)
    Kx = axis_vector[0]
    Ky = axis_vector[1]
    Kz = axis_vector[2]

    R_k = np.array([
        [Kx*Kx*(1-np.cos(theta))+np.cos(theta), Kx*Ky*(1-np.cos(theta))-Kz*np.sin(theta), Kx*Kz*(1-np.cos(theta))+Ky*np.sin(theta)],
        [Kx*Ky*(1-np.cos(theta))+Kz*np.sin(theta), Ky*Ky*(1-np.cos(theta))+np.cos(theta), Ky*Kz*(1-np.cos(theta))-Kx*np.sin(theta)],
        [Kx*Kz*(1-np.cos(theta))-Ky*np.sin(theta), Ky*Kz*(1-np.cos(theta))+Kx*np.sin(theta), Kz*Kz*(1-np.cos(theta))+np.cos(theta)]
    ])

    if not p:
        return R_k

    return rotate_R(R_k, p)


def R2axis_angle(R):
    """
    旋转矩阵转换为轴角
    ------------------------
    Rotation matrix converted to axis angle
    """
    theta = np.arccos((mat.trace(R)-1)/2)
    cpm = (R-mat.transpose(R))/(2*np.sin(theta))
    vector = mat.cpm2vec(cpm)
    return vector, theta


def euler_zyx2quaternion(theta_z, theta_y, theta_x):
    '''
    欧拉角(Z-Y-X)转换为四元数
    ---------------------------
    Euler angles (Z-Y-X) to quaternions
    '''
    w = np.sin(theta_x / 2) * np.sin(theta_y / 2) * np.sin(theta_z / 2) + \
        np.cos(theta_x / 2) * np.cos(theta_y / 2) * np.cos(theta_z / 2)
    x = np.sin(theta_x / 2) * np.cos(theta_y / 2) * np.cos(theta_z / 2) - \
        np.cos(theta_x / 2) * np.sin(theta_y / 2) * np.sin(theta_z / 2)
    y = np.cos(theta_x / 2) * np.sin(theta_y / 2) * np.cos(theta_z / 2) + \
        np.sin(theta_x / 2) * np.cos(theta_y / 2) * np.sin(theta_z / 2)
    z = np.cos(theta_x / 2) * np.cos(theta_y / 2) * np.sin(theta_z / 2) - \
        np.sin(theta_x / 2) * np.sin(theta_y / 2) * np.cos(theta_z / 2)

    return np.array([x,y,z,w])


def axis_angle2quaternion(axis_vector, theta):
    '''
    轴角转换为四元数
    ----------------------
    Axis angle to quaternion
    '''
    axis_vector = dproc.normalize(axis_vector, normType=params.norm_L2)

    w = np.cos(theta / 2)
    x = axis_vector[0] * np.sin(theta / 2)
    y = axis_vector[1] * np.sin(theta / 2)
    z = axis_vector[2] * np.sin(theta / 2)

    return np.array([w,x,y,z])


def quaternion2axis_angle(q):
    """
    四元数转换为轴角
    -----------------------
    Quaternion to axis angle
    """
    w, x, y, z = q

    theta = 2*np.arccos(w)
    a, b, c = np.array([x,y,z]) / np.sin(theta / 2)

    vector = np.array([a,b,c])
    return vector, theta


def R2quaternion(R):
    '''
    旋转矩阵转换为四元数
    ----------------------
    rotation matrix to quaternion
    '''
    w = np.sqrt(1+R[0,0]+R[1,1]+R[2,2]) / 2
    x = (R[2,1] - R[1,2]) / (4*w)
    y = (R[0,2] - R[2,0]) / (4*w)
    z = (R[1,0] - R[0,1]) / (4*w)

    return np.array([w,x,y,z])


def quaternion2R(q):
    """
    四元数转换为旋转矩阵
    ------------------------
    Quaternion to rotation matrix
    """
    w,x,y,z = q

    R = np.array([[1-2*y*y-2*z*z, 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*x*x-2*z*z, 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*x*x-2*y*y]])

    return R


@nb.njit()
def rotate_quaternion(q, p):
    '''
    利用四元数进行旋转
    -----------------------
    Using quaternions for rotation
    '''
    p = np.concatenate((np.array([0]), p))

    q_inv = complex.quaternion_inverse(q)
    q_mul_p = complex.quaternion_mul(q, p)
    p_rotated = complex.quaternion_mul(q_mul_p, q_inv)

    return p_rotated[1:]



def mirror(omega, p):
    """
    将向量p沿着与单位向量omega垂直的方向做镜像变换
    ----------------------------------------------
    The vector p is mirrored in the direction perpendicular to the unit vector Omega
    """
    assert omega.shape == (3,) and p.shape==(3,)
    omega = dproc.normalize(omega, normType=params.norm_L2)
    omega = omega[np.newaxis, :]
    p = p[np.newaxis, :]

    I = np.eye(3)
    H = I - 2*np.matmul(mat.transpose(omega), omega)
    return np.matmul(H, mat.transpose(p)).squeeze()

