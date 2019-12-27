import numpy as np




def DH(alpha1, a1, theta2, d2):
    """
    正运动学求解，DH参数法
    ---------------------------
    Forward kinematics solution, DH parameter method
    """
    T = np.array(
        [[np.cos(theta2), -np.sin(theta2), 0, a1],
         [np.sin(theta2)*np.cos(alpha1), np.cos(theta2)*np.cos(alpha1), -np.sin(alpha1), -np.sin(alpha1)*d2],
         [np.sin(theta2)*np.sin(alpha1), np.cos(theta2)*np.sin(alpha1), np.cos(alpha1), np.cos(alpha1)*d2],
         [0, 0, 0, 1]]
    )

    return T


