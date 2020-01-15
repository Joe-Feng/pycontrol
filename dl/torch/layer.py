import torch




def sigmoid(x):
    return 1 / (1 + torch.exp(-1*x))


def tanh(x):
    return (torch.exp(x) - torch.exp(-1*x)) / (torch.exp(x) + torch.exp(-1*x))


def relu(x, inplace=True):
    if inplace:
        x[x < 0] = 0
        return x
    else:
        d = torch.clone(x)
        d[d < 0] = 0
        return d


def softmax(x):
    exps = torch.exp(x-torch.max(x))
    return exps / torch.sum(exps)



