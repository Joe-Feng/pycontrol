import numpy as np
from pycontrol import params



def normalize(vector):
    '''
    规范化
    '''
    norm = np.sqrt(np.sum(np.power(vector, 2)))
    if norm == 0:
        norm += params.sys_min
    return vector / norm


