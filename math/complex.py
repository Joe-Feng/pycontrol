from pycontrol.matrix import judge
import numpy as np



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


def quaternion_modulus(q):
    '''
    四元数的模
    ---------------
    Modulus of Quaternion
    '''
    w,x,y,z = q

    return np.sqrt(w*w + x*x + y*y + z*z)

def quaternion_conjugate(q):
    '''
    四元数的共轭
    -----------------
    Conjugation of quaternions
    '''
    w, x, y, z = q

    return np.array([w, -x, -y, -z])

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

