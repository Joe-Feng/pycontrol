from pycontrol import dl
import numpy as np
import torch
import matplotlib.pyplot as plt



def plot(x):
    fig = plt.figure()

    fig.add_subplot(221)
    plt.title('relu')
    y = show_relu(x)
    plt.plot(x, y)

    fig.add_subplot(222)
    plt.title('sigmoid')
    y = show_sigmoid(x)
    plt.plot(x, y)

    fig.add_subplot(223)
    plt.title('tanh')
    y = show_tanh(x)
    plt.plot(x, y)

    plt.show()

def show_relu(x):
    y = dl.relu(x)
    return y.numpy()

def show_sigmoid(x):
    y = dl.sigmoid(x)
    return y.numpy()

def show_tanh(x):
    y = dl.tanh(x)
    return y.numpy()


if __name__ == '__main__':
    x = torch.arange(-5, 5, step=0.1, dtype=torch.float64)
    plot(x)

