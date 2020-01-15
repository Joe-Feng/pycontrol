import torch
from torch.optim.optimizer import Optimizer, required



class SGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, nesterov=nesterov)

        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad_data = p.grad.data

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad_data).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1-dampening, grad_data)

                    if nesterov:
                        grad_data = grad_data.add(momentum, buf)
                    else:
                        grad_data = buf


                p.data.add_(-1*lr, grad_data)

