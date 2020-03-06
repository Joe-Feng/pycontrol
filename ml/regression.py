import numpy as np
from pycontrol import params




def least_squares(A, b=None, method=params.LS_default):
    """
    最小二乘法求解min||Ax=b||或min||Ax=0||
    -------------------------------------------
    Least squares find min||Ax=b|| or min||Ax=0||
    """
    if b is None:
        method = params.LS_svd

    if method == params.LS_default:
        estimated = np.linalg.inv(np.matmul(A.T, A))
        estimated = np.matmul(estimated, A.T)
        estimated = np.matmul(estimated, b)

    elif method == params.LS_svd:
        if b is not None:
            y = np.zeros(shape=(A.shape[1], 1))
            U, D, V_T = np.linalg.svd(A)        # SVD
            b_hat = np.matmul(U.T, b)           # b_hat = U_T*b
            for i in range(y.shape[0]):         # y[i] = b_hat[i] / d[i]
                y[i][0] = b_hat[i][0] / D[i]

            estimated = np.matmul(V_T.T, y)     # x = V*y


    return estimated