import numpy as np
import matplotlib.pyplot as plt
import cv2



def normal_distribution1d(x, sigma=1, mu=0):
    """
    一维随机变量正态分布
    ----------------------------
    Normal distribution of one-dimensional random variables
    """
    norm = \
        1 / (np.sqrt(2*np.pi) * sigma) * \
        np.exp(-np.power(x-mu, 2) / (2*sigma*sigma))
    return norm


def normal_distribution2d(x, y, sigma1=1.0, sigma2=1.0, mu1=0, mu2=0, rho=0):
    """
    二维随机变量正态分布
    ----------------------------
    Normal distribution of two-dimensional random variables
    """
    norm = \
        1 / (2*np.pi*sigma1*sigma2*np.sqrt(1-rho*rho)) * \
        np.exp(-1/(2*(1-rho*rho)) * (
            np.power(x-mu1, 2) / (sigma1*sigma1) -
            2*rho*(x-mu1)*(y-mu2) / (sigma1*sigma2) +
            np.power(y-mu2, 2) / (sigma2*sigma2)
        ))
    return norm


