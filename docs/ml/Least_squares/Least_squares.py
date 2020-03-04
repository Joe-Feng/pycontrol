import numpy as np
from pycontrol import mat
import matplotlib.pyplot as plt
import time



# true value
ar = 10.0
br = 5.0
cr = 2.0

N = 100
sigma = 0.5

np.random.seed(123)

def gen_dataset():
    x_data = np.arange(0, N) / N
    y_data = ar*x_data*x_data + br*x_data + cr + \
             np.random.normal(scale=sigma, size=(N,))

    return x_data, y_data


def plot(x_data, y_data, estimated):
    ae, be, ce = estimated[0][0], estimated[1][0], estimated[2][0]
    y_pred = ae * x_data * x_data + be * x_data + ce

    legend = ['pred']

    plt.figure()
    plt.plot(x_data, y_pred, c='red')

    plt.scatter(x_data, y_data, c='blue')

    plt.title('Least_squares')
    plt.legend(legend)
    plt.show()


def solve(x_data, y_data):
    X = x_data[..., np.newaxis]
    Y = y_data[..., np.newaxis]
    A = np.zeros(shape=[N,3])

    A[:, 0:1] = np.power(X, 2)
    A[:, 1:2] = X
    A[:, 2:3] = 1
    b = Y

    estimated = np.linalg.inv(np.matmul(A.T, A))
    estimated = np.matmul(estimated, A.T)
    estimated = np.matmul(estimated, b)

    return estimated



if __name__ == '__main__':
    x_data, y_data = gen_dataset()
    estimated = solve(x_data, y_data)
    print("estimated:", estimated.squeeze())
    plot(x_data, y_data, estimated)



