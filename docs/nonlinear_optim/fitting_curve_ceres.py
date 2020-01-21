from pycontrol import ceres
import numpy as np
import torch
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



class CURVE_FITTING_COST():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def operator(self, abc):
        self.residual = \
            self.y - torch.exp(
                abc[0]*self.x*self.x + abc[1]*self.x + abc[2]
            )



if __name__ == '__main__':
    x_data, y_data = gen_dataset()
    abc = [ae, be, ce]

    # 构建最小二乘问题
    # Constructing the least square problem
    problem = ceres.Problem()
    problem.AddResidualBlock(
        CURVE_FITTING_COST(
            x_data, y_data
        ),
        None,
        abc
    )

    # 配置求解器
    # Configure solver
    options = ceres.Options()
    options.linear_solver_type = ceres.DENSE_NORMAL_CHOLESKY
    options.minimizer_progress_to_stdout = True

    start = time.time()
    solver = ceres.Solver(options, problem)
    print('time: ', time.time() - start)

    print('real abc: {}'.format([ar, br, cr]))
    print('estimated abc: {}'.format(solver.estimated_params))



