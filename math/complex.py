from pycontrol.matrix import judge
import numpy as np
import numba as nb


def quaternion_add(q1, q2):
    '''
    四元数加法
    ---------------
    Quaternion addition
    '''
    if not judge.is_ndarray(q1):
        q1 = np.array(q1)
    if not judge.is_ndarray(q2):
        q2 = np.array(q2)

    return q1 + q2


def quaternion_sub(q1, q2):
    '''
    四元数减法
    -------------
    Quaternion subtraction
    '''
    if not judge.is_ndarray(q1):
        q1 = np.array(q1)
    if not judge.is_ndarray(q2):
        q2 = np.array(q2)

    return q1 - q2


@nb.njit()
def quaternion_mul(q1, q2):
    '''
    四元数乘法
    --------------
    Quaternion multiplication
    '''
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return -1*np.array([w,x,y,z])


@nb.njit()
def quaternion_modulus(q):
    '''
    四元数的模
    ---------------
    Modulus of Quaternion
    '''
    w,x,y,z = q

    return np.sqrt(w*w + x*x + y*y + z*z)


@nb.njit()
def quaternion_conjugate(q):
    '''
    四元数的共轭
    -----------------
    Conjugation of quaternions
    '''
    w, x, y, z = q

    return np.array([w, -x, -y, -z])


@nb.njit()
def quaternion_inverse(q):
    '''
    四元数的逆
    ------------------
    The inverse of Quaternion
    '''
    q_conj = quaternion_conjugate(q)
    q_mod = quaternion_modulus(q)
    q_inv = q_conj / (q_mod*q_mod)

    return q_inv



def complex_add(x1, y1, x2, y2):
    Re = x1+x2
    Im = y1+y2
    return Re, Im

def complex_sub(x1, y1, x2, y2):
    Re = x1-x2
    Im = y1-y2
    return Re, Im

def complex_mul(x1, y1, x2, y2):
    Re = x1*x2 - y1*y2
    Im = x1*y2 + x2*y1
    return Re, Im

def complex_div(x1, y1, x2, y2):
    Re = (x1*x2 + y1*y2) / (x2*x2 + y2*y2)
    Im = (x2*y1 - x1*y2) / (x2*x2 + y2*y2)
    return Re, Im


def complex_modulus(x, y):
    m = np.sqrt(x*x + y*y)
    return m


def complex_conj(x, y):
    return x, -y


def complex_argz(x, y):
    """
    求辐角的主值
    ------------------
    Calculate the principal value of the angle of a complex number
    """
    if x > 0:
        argz = np.arctan(y/x)
    elif x == 0 and y > 0:
        argz = np.pi / 2
    elif x < 0 and y >= 0:
        argz = np.arctan(y / x) + np.pi
    elif x < 0 and y < 0:
        argz = np.arctan(y / x) - np.pi
    else:
        argz = -np.pi / 2

    return argz



def complex_power(x, y, n):
    """
    复数的n次幂 (棣莫弗公式)
    ------------
    nth power of complex number (De Moivre formula)
    """
    r = np.sqrt(x*x + y*y)
    theta = complex_argz(x, y)
    result_x = np.power(r,n) * np.cos(n*theta)
    result_y = np.power(r,n) * np.sin(n*theta)
    return result_x, result_y


