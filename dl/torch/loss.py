import torch
import torch.nn as nn
from pycontrol.dl.torch import layer
from pycontrol.dl.torch import utils as dlutils




class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, outputs, targets):
        return 0.5 * torch.mean(torch.sum(torch.pow((outputs - targets), 2), dim=1))


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, outputs, targets):
        outputs = layer.softmax(outputs)
        targets = torch.unsqueeze(targets, dim=1)
        outputs = torch.gather(outputs, 1, targets)
        return -1*torch.mean(torch.log(torch.clamp(outputs, 1e-10, 1.0)))


