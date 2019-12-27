import numpy as np
from pycontrol import params



def is_ndarray(matrix):
    '''
    判断是否为np.ndarray
    '''
    if not isinstance(matrix, np.ndarray):
        return False
    return True


def is_square_matrix(matrix):
    '''
    判断是否为方阵
    '''
    h = matrix.shape[0]
    w = matrix.shape[1]
    if h==w:
        return True
    else:
        return False


# TODO 修改判断方法
def is_orthogona_matrix(matrix):
    '''
    判断是否为正交阵
    '''
    if not is_ndarray(matrix):
        matrix = np.array(matrix)

    if not is_square_matrix(matrix):
        raise Exception("rotate matrix must be a square matrix")

    M = np.matmul(matrix, matrix.transpose())
    h = matrix.shape[0]
    w = matrix.shape[1]
    for i in range(h):
        for j in range(w):
            if i==j:
                if M[i,j] < params.close_to_1 or M[i,j] > 1:
                    return False
            else:
                if M[i,j] != 0:
                    return False
    return True

