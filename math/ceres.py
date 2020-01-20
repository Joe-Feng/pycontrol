import numpy as np
import torch


DENSE_NORMAL_CHOLESKY = 0


class Problem():
    def AddResidualBlock(self, CostClass, kernel, estimated_params):
        if not isinstance(estimated_params, np.ndarray):
            self.estimated_params = np.array(estimated_params)

        self.cost = CostClass
        self.var_dict = self.cost.__dict__
        for key in self.var_dict.keys():
            if not isinstance(self.var_dict[key], np.ndarray):
                self.var_dict[key] = np.array(self.var_dict[key])
            self.var_dict[key] = torch.from_numpy(self.var_dict[key]).float().requires_grad_(False)
            self.n_data = self.var_dict[key].size()[0]

        self.estimated_params = torch.from_numpy(self.estimated_params).float().requires_grad_(True)



class Options():
    def __init__(self):
        self.linear_solver_type = DENSE_NORMAL_CHOLESKY
        self.minimizer_progress_to_stdout = True
        self.max_num_iterations = 10


class Solver():
    def __init__(self, Options, Problem):
        self.options = Options
        self.problem = Problem
        self.estimated_params = self.problem.estimated_params

        self.solve()

    def solve(self):
        last_cost = 0

        for iter in range(self.options.max_num_iterations):
            self.problem.cost.operator(self.estimated_params)
            residual = self.problem.cost.residual

            H = torch.zeros(size=(3, 3))
            b = torch.zeros(size=(3, 1))
            cost = 0

            for i in range(self.problem.n_data):
                residual[i].backward(retain_graph=True)
                J = self.estimated_params.grad
                J = torch.unsqueeze(J, dim=1)

                error = residual[i]
                H += torch.matmul(J, J.transpose(1,0))
                b += -error * J

                cost += error * error

                self.zero_grad()

            dx = torch.matmul(torch.inverse(H), b).squeeze()

            for i in range(dx.size()[0]):
                if torch.isnan(dx[i]):
                    print('result is nan!')
                    break

            if iter > 0 and cost >= last_cost:
                print('cost: {}, last cost: {}, break.'.format(cost, last_cost))
                break

            self.estimated_params.data.add_(dx)

            last_cost = cost

            print('total cost: {}, update: {}, estimated params: {}'.format(
                cost, dx.detach().numpy(), self.estimated_params.detach().numpy()
            ))

        self.estimated_params = self.estimated_params.detach().numpy()

    def zero_grad(self):
        if self.estimated_params.grad is not None:
            self.estimated_params.grad.detach_()
            self.estimated_params.grad.zero_()


