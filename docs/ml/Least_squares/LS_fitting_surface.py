import numpy as np
from pycontrol import ml, params
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import time



# true value
A = 6.0
B = 3.0
C = 3.0


N = 100
sigma = 0.5

np.random.seed(123)

def gen_dataset():
    data = np.random.randint(-N, N, (N,2)) / N
    x_data = data[:,0]
    y_data = data[:,1]
    z_data = A*x_data*x_data + B*y_data*y_data + C + \
             np.random.normal(scale=sigma, size=(N,))

    return x_data, y_data, z_data


def plot(x_data, y_data, z_data, estimated):
    Ae, Be, Ce = estimated[0][0], estimated[1][0], estimated[2][0]

    ax = plt.axes(projection='3d')

    z_data = np.sqrt(z_data)
    ax.scatter3D(x_data, y_data, z_data, color='blue', depthshade=False)

    x_data, y_data = np.meshgrid(x_data, y_data)
    z_pred = Ae * x_data*x_data + Be * y_data*y_data + Ce
    z_pred = np.sqrt(z_pred)
    ax.plot_surface(x_data, y_data, z_pred, cmap='rainbow', shade=False)

    plt.title('LS_fitting_surface')
    plt.show()


def solve(x_data, y_data, z_data):
    X = x_data[..., np.newaxis]
    Y = y_data[..., np.newaxis]
    Z = z_data[..., np.newaxis]
    A = np.zeros(shape=[N,3])

    A[:, 0:1] = X*X
    A[:, 1:2] = Y*Y
    A[:, 2:3] = 1
    b = Z

    estimated = ml.least_squares(A, b, params.LS_svd)

    return estimated



if __name__ == '__main__':
    x_data, y_data, z_data = gen_dataset()
    estimated = solve(x_data, y_data, z_data)
    print("estimated:", estimated.squeeze())
    plot(x_data, y_data, z_data, estimated)



