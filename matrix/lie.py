import numpy as np
from pycontrol.matrix import transform
from pycontrol.matrix import matrix as mat
from pycontrol.math import complex
from pycontrol import params
import numba as nb


def SO3_R(R):
    """
    从旋转矩阵构造李代数
    ----------------------
    Constructing Lie algebra from rotation matrix
    """
    return transform.R2quaternion(R)


def SO3_q(q):
    """
    从四元数构造李代数
    ----------------------
    Constructing Lie algebra from quaternion
    """
    return q


def SO3_matrix(SO3):
    return transform.quaternion2R(SO3)


def SO3_log(SO3):
    """
    使用对数映射获得李代数
    ------------------------
    Using logarithmic mapping to obtain Lie algebra
    """
    # # 方法一
    # vector, angle = transform.quaternion2axis_angle(SO3)
    # return angle*vector

    # 方法二
    n = np.sqrt(np.sum(np.power(SO3[1:], 2)))
    return 2*np.arctan(n / SO3[0]) / n * SO3[1:]


def SO3_hat(so3):
    """
    向量到反对称矩阵
    --------------------
    Vector to antisymmetric matrix
    """
    return mat.cross_product_matrix(so3)


def SO3_vee(so3_hat):
    """
    反对称矩阵到向量
    -------------------
    antisymmetric matrix to vector
    """
    return mat.cpm2vec(so3_hat)


def SO3_inverse(SO3):
    """
    求SO(3)的逆
    ------------------
    Finding the inverse of SO(3)
    """
    return complex.quaternion_inverse(SO3)


def SO3_mul_SO3(SO3_1, SO3_2):
    """
    SO(3)与SO(3)乘法
    ------------------
    SO(3) and SO(3) multiplication
    """
    return complex.quaternion_mul(SO3_1, SO3_2)


@nb.njit()
def SO3_mul_p(SO3, p):
    """
    SO(3)与点的乘法
    --------------------
    SO(3) and point multiplication
    """
    return transform.rotate_quaternion(SO3, p)


def so3_exp_update(update_so3, R=None):
    """
    增量扰动模型的更新
    ---------------------
    Update of incremental disturbance model
    """
    # 李代数的模是theta
    theta = np.sqrt(np.sum(np.power(update_so3, 2)))
    if theta == 0:
        theta = params.sys_min

    # # 方法一
    # update_R = transform.rotate_axis_angle(update_so3/theta, theta)
    # if np.shape(R) == ():
    #     return update_R
    # return np.matmul(update_R, R)

    # 方法二
    update_q = transform.axis_angle2quaternion(update_so3/theta, theta)
    if np.shape(R) == ():
        return update_q

    q = transform.R2quaternion(R)
    return SO3_mul_SO3(update_q, q)



def SE3_Rt(R, t):
    """
    从旋转矩阵和平移构造李代数
    ----------------------
    Constructing Lie algebra from rotation matrix and translation
    """
    SO3 = SO3_R(R)
    return np.append(t, SO3)


def SE3_qt(q, t):
    """
    从四元数和平移构造李代数
    ----------------------
    Constructing Lie algebra from quaternion and translation
    """
    SO3 = SO3_q(q)
    return np.append(t, SO3)


def SE3_matrix(SE3):
    R = SO3_matrix(SE3[3:])
    t = SE3[:3]
    t = t[np.newaxis, :]
    t = t.transpose()
    conc = np.concatenate((R, t), axis=1)
    return np.concatenate((conc, [[0, 0, 0, 1]]), axis=0)


def se3_J(vector, theta):
    vector_T = mat.transpose(vector)
    vector_hat = mat.cross_product_matrix(np.squeeze(vector))
    I = np.eye(3)

    J = (np.sin(theta)/theta)*I + \
        (1-np.sin(theta)/theta)*np.matmul(vector,vector_T) + \
        (1-np.cos(theta))/theta*vector_hat

    return J


def SE3_log(SE3):
    """
    使用对数映射获得李代数
    ------------------------
    Using logarithmic mapping to obtain Lie algebra
    """

    # # 方法一
    # vector, angle = transform.quaternion2axis_angle(SE3[3:])
    #
    # vector = mat.transpose(np.expand_dims(vector, 0))
    # t = mat.transpose(np.expand_dims(SE3[:3], 0))
    #
    # J = se3_J(vector, angle)
    # rho = np.matmul(mat.inverse(J), t)
    # return np.append(rho, angle * vector)


    # 方法二
    omega = SO3_log(SE3[3:])
    t = SE3[:3]
    theta = np.sqrt(np.sum(np.power(omega, 2)))
    omega_hat = SO3_hat(omega)

    V_inv = np.eye(3) - 0.5 * omega_hat + \
            (1 - theta * np.cos(theta / 2) / (2 * np.sin(theta / 2))) / (theta ** 2) * \
            np.matmul(omega_hat, omega_hat)

    epsilon = np.matmul(V_inv, np.expand_dims(t, 0).transpose())

    return np.append(epsilon.squeeze(), omega)



def SE3_hat(SE3):
    """
    向量到反对称矩阵
    --------------------
    Vector to antisymmetric matrix
    """
    cpm = mat.cross_product_matrix(SE3[3:])

    rho = SE3[:3]
    rho = rho[np.newaxis, :]
    rho = rho.transpose()

    conc = np.concatenate((cpm, rho), axis=1)
    se3_hat = np.concatenate((conc, [[0, 0, 0, 0]]), axis=0)
    return se3_hat


def SE3_vee(SE3_hat):
    """
    反对称矩阵到向量
    -------------------
    antisymmetric matrix to vector
    """
    R = SE3_hat[0:3, 0:3]
    rho = SE3_hat[0:3, 3:4]

    vee = mat.cpm2vec(R)
    rho = np.squeeze(rho)

    return np.append(rho, vee)


def SE3_inverse(SE3):
    """
    求SE(3)的逆
    ------------------
    Finding the inverse of SE(3)
    """
    SO3_inv = SO3_inverse(SE3[3:])
    t_inv = SO3_mul_p(SO3_inv, -1*SE3[:3])
    return np.append(t_inv, SO3_inv)


def SE3_mul_SE3(SE3_1, SE3_2):
    """
    SE(3)与SE(3)乘法
    ------------------
    SE(3) and SE(3) multiplication
    """
    r = SO3_mul_SO3(SE3_1[3:], SE3_2[3:])
    t = SE3_1[:3] + SO3_mul_p(SE3_1[3:], SE3_2[:3])

    return np.append(t, r)


@nb.njit()
def SE3_mul_p(SE3, p):
    """
    SE(3)与点的乘法
    ------------------
    SE(3) and point multiplication
    """
    return SE3[:3] + SO3_mul_p(SE3[3:], p)


def se3_exp_update(update_se3, R=None, t=None):
    """
    增量扰动模型的更新
    ---------------------
    Update of incremental disturbance model
    """
    epsilon = update_se3[:3]
    omega = update_se3[3:]

    so3_update = so3_exp_update(omega)
    omega_hat = SO3_hat(omega)
    theta = np.sqrt(np.sum(np.power(omega, 2)))
    if theta == 0:
        theta = params.sys_min

    V = np.eye(3) + \
        (1-np.cos(theta)) / (theta**2)*omega_hat + \
        (theta-np.sin(theta)) / (theta**3)*(np.matmul(omega_hat,omega_hat))


    update_t = np.matmul(V, np.expand_dims(epsilon, 0).transpose())
    if np.shape(R) == ():
        return np.append(update_t.squeeze(), so3_update)


    SE3 = SE3_Rt(R, t)
    update_SE3 = np.append(epsilon, so3_update)
    return SE3_mul_SE3(update_SE3, SE3)


