import numpy as np
from pycontrol import mat
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import time



# true value
A = 6.0
B = 3.0
C = 2.0


N = 100
sigma = 1.0

np.random.seed(123)

def gen_dataset():
    data = np.random.randint(0, N, (N,2)) / N
    x_data = data[:,0]
    y_data = data[:,1]
    z_data = A*x_data + B*y_data + C + \
             np.random.normal(scale=sigma, size=(N,))

    return x_data, y_data, z_data


def plot(x_data, y_data, z_data, estimated):
    Ae, Be, Ce = estimated[0][0], estimated[1][0], estimated[2][0]

    ax = plt.axes(projection='3d')

    ax.scatter3D(x_data, y_data, z_data, color='blue', depthshade=False)

    x_data, y_data = np.meshgrid(x_data, y_data)
    z_pred = Ae * x_data + Be * y_data + Ce
    ax.plot_surface(x_data, y_data, z_pred, cmap='rainbow', shade=False)

    plt.title('LS_fitting_plane')
    plt.show()


def solve(x_data, y_data, z_data):
    X = x_data[..., np.newaxis]
    Y = y_data[..., np.newaxis]
    Z = z_data[..., np.newaxis]
    A = np.zeros(shape=[N,3])
    y = np.zeros(shape=(A.shape[1], 1))

    A[:, 0:1] = X
    A[:, 1:2] = Y
    A[:, 2:3] = 1
    b = Z

    U, D, V_T = np.linalg.svd(A)  # SVD
    b_hat = np.matmul(U.T, b)  # b_hat = U_T*b
    for i in range(y.shape[0]):  # y[i] = b_hat[i] / d[i]
        y[i][0] = b_hat[i][0] / D[i]

    estimated = np.matmul(V_T.T, y)  # x = V*y

    return estimated



if __name__ == '__main__':
    x_data, y_data, z_data = gen_dataset()
    estimated = solve(x_data, y_data, z_data)
    print("estimated:", estimated.squeeze())
    plot(x_data, y_data, z_data, estimated)



