import numpy as np
from pycontrol import mat
import matplotlib.pyplot as plt
import time


# true value
ar = 1.0
br = 2.0
cr = 1.0

# estimated value
ae = 2.0
be = -1.0
ce = 5.0

N = 100
sigma = 1.0
inv_sigma = 1.0 / sigma

np.random.seed(123)

def gen_dataset():
    x_data = np.arange(0, N) / 100.0
    y_data = np.exp(ar*x_data*x_data + br*x_data + cr) + \
             np.random.normal(scale=sigma, size=(N,))

    return x_data, y_data

def plot(x_data, ae, be, ce):
    y_true = np.exp(ar * x_data * x_data + br * x_data + cr)
    y_pred = np.exp(ae * x_data * x_data + be * x_data + ce)

    legend = ['true', 'pred']

    plt.figure()
    plt.plot(x_data, y_true, c='green')
    plt.plot(x_data, y_pred, c='red')

    y_true += np.random.normal(scale=sigma, size=(N,))
    plt.scatter(x_data, y_true, c='blue')

    plt.title('fitting_curve')
    plt.legend(legend)
    plt.show()


def update(x_data, y_data, params):
    ae, be, ce = params
    n_iters = 10
    last_cost = 0

    for iter in range(n_iters):
        H = np.zeros(shape=(3, 3))
        b = np.zeros(shape=(3, 1))
        J = np.zeros(shape=(3, 1))
        cost = 0

        # 方法一
        for i in range(N):
            x, y = x_data[i], y_data[i]
            error = y - np.exp(ae * x * x + be * x + ce)

            J[0][0] = -x * x * np.exp(ae * x * x + be * x + ce)
            J[1][0] = -x * np.exp(ae * x * x + be * x + ce)
            J[2][0] = -np.exp(ae * x * x + be * x + ce)

            H += inv_sigma * inv_sigma * np.matmul(J, J.transpose())
            b += -inv_sigma * inv_sigma * error * J

            cost += error * error


        # # 方法二
        # error = y_data - np.exp(ae * x_data * x_data + be * x_data + ce)
        #
        # diff = []
        # d = -x_data * x_data * np.exp(ae * x_data * x_data + be * x_data + ce)
        # diff.append(d)
        # d = -x_data * np.exp(ae * x_data * x_data + be * x_data + ce)
        # diff.append(d)
        # d = -np.exp(ae * x_data * x_data + be * x_data + ce)
        # diff.append(d)
        # diff = np.array(diff)
        #
        # H = inv_sigma * inv_sigma * np.matmul(diff, diff.transpose())
        # b = -inv_sigma * inv_sigma * np.sum(error * diff, axis=1, keepdims=True)
        #
        # cost = np.sum(np.square(error))

        ########################################################################################

        dx = np.matmul(mat.inverse(H), b)

        if np.isnan(dx.any()):
            print('result is nan!')
            break

        if iter > 0 and cost >= last_cost:
            print('cost: {}, last cost: {}, break.'.format(cost, last_cost))
            break

        ae += dx[0][0]
        be += dx[1][0]
        ce += dx[2][0]

        last_cost = cost

        print('total cost: {}, update: {}, estimated params '
              'ae: {}, be: {}, ce: {}'.format(
            cost, dx.transpose(), ae, be, ce
        ))

    return ae, be, ce


if __name__ == '__main__':
    x_data, y_data = gen_dataset()
    plot(x_data, ae, be, ce)

    start = time.time()
    ae, be, ce = update(x_data, y_data, [ae,be,ce])
    print('time: ', time.time() - start)

    print('real abc = {}, {}, {}'.format(ar, br, cr))
    print('estimated abc = {}, {}, {}'.format(ae, be, ce))

    plot(x_data, ae, be, ce)




