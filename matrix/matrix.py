import numpy as np
from pycontrol.matrix import judge
from pycontrol.data_science import dproc



def cross_product(vector_a, vector_b):
    '''
    叉乘
    '''
    a1, a2, a3 = vector_a
    b1, b2, b3 = vector_b

    C = np.array([[a2*b3 - a3*b2],
                  [a3*b1 - a1*b3],
                  [a1*b2 - a2*b1]])

    return C


def cross_product_matrix(vector):
    '''
    求向量的叉乘矩阵(反对称矩阵)
    -------------------------------
    Finding the cross product matrix of vector (antisymmetric matrix)
    '''
    a1, a2, a3 = vector

    A = np.array([[0, -a3, a2],
                  [a3, 0, -a1],
                  [-a2, a1, 0]])

    return A


def cpm2vec(matrix):
    """
    由叉乘矩阵求向量
    ------------------------
    Finding vector from cross product matrix
    """
    vector = np.array([matrix[2][1], matrix[0][2], matrix[1][0]])
    return vector


def trace(matrix):
    '''
    求方阵的迹
    '''
    if not judge.is_square_matrix(matrix):
        raise Exception("only square matrix can compute trace")

    trace = 0
    h = matrix.shape[0]
    for i in range(h):
        trace += matrix[i,i]

    return trace


# TODO 自己实现
def transpose(matrix):
    '''
    矩阵转置
    -----------------------
    matrix transpose
    '''
    return matrix.transpose()


def sum(matrix):
    return matrix.sum()


def determinant(matrix):
    '''
    求方阵的行列式
    '''
    if not judge.is_square_matrix(matrix):
        raise Exception("only square matrix can compute determinant")

    return np.linalg.det(matrix)



def power_det(matrix, p):
    """
    矩阵p次幂的行列式
    ---------------------
    Determinant of P power of matrix
    """
    m_det = determinant(matrix)
    power_det = np.power(m_det, p)
    return power_det


def adjoint_det(matrix):
    """
    矩阵的伴随矩阵的行列式
    ------------------------
    Determinant of adjoint matrix of matrix
    """
    h = matrix.shape[0]

    m_det = determinant(matrix)
    a_det = np.power(m_det, h-1)
    return a_det





def adjoint(matrix):
    '''
    求矩阵的伴随矩阵
    --------------------
    Find the adjoint matrix of a matrix
    '''
    h = matrix.shape[0]
    w = matrix.shape[1]
    adj_M = np.zeros(shape=[h,w])

    for i in range(h):
        for j in range(w):
            k = 1 if (i+j) % 2 == 0 else -1
            m = matrix.copy()
            m = np.delete(m, i, axis=0)
            m = np.delete(m, j, axis=1)
            det = determinant(m)
            adj_M[i,j] = k*det

    return adj_M.transpose()



def inverse_w_adjoint(matrix):
    '''
    求矩阵的逆：利用伴随矩阵
    -----------------------------
    Finding the inverse of a matrix: using an adjoint matrix
    '''
    if not judge.is_square_matrix(matrix):
        raise Exception("only square matrix can compute inverse")

    det = determinant(matrix)
    if det == 0:
        raise Exception('singular matrix has no inverse matrix')

    adj_M = adjoint(matrix)

    return adj_M / det




def inverse_w_schmidt_qr(matrix):
    '''
    求矩阵的逆：利用施密特QR分解
    -------------------------------
    Finding the inverse of matrix: using Schmidt QR decomposition
    '''
    Q, R = schmidt_qr(matrix)
    inv_Q = Q.transpose()
    inv_R = inverse(R)

    return np.matmul(inv_R, inv_Q)


def inverse(matrix):
    """
    求逆矩阵
    """
    return np.linalg.inv(matrix)


def inverse_adjoint(matrix):
    """
    求矩阵的伴随矩阵的逆
    -----------------------
    Finding the inverse of adjoint matrix of matrix
    """
    # TODO 完成
    pass


# TODO 自己实现
def eigen(matrix):
    '''
    求矩阵的特征值和特征向量
    ------------------------------
    Finding eigenvalues and eigenvectors of matrices
    '''
    return np.linalg.eig(matrix)



def schmidt_ort(matrix):
    '''
    施密特正交化
    ------------------
    schmidt orthogonalization
    '''
    w = matrix.shape[1]
    m_copy = matrix.copy()
    matrix = matrix.copy()

    for k in range(w):
        for i in range(k):
            matrix[:,k] -= \
                np.matmul(m_copy[:,k], matrix[:,i])/np.matmul(matrix[:,i], matrix[:,i]) * \
                matrix[:,i]
        matrix[:,k] = dproc.normalize(matrix[:,k])

    return matrix

def schmidt_qr(matrix):
    '''
    QR分解：schmidt正交化方法
    ----------------------------
    QR decomposition: schmidt orthogonalization method
    '''
    Q = schmidt_ort(matrix)
    R = np.matmul(Q.transpose(), matrix)

    return Q, R

# TODO 完成
def householder_qr():
    pass

# TODO 完成
def givens_qr():
    pass

# TODO 完成
def cholesky():
    pass


