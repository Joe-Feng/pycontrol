from pycontrol import math, mat
import numpy as np



# 由轴角旋转得到四元数
# Get quaternions from the rotation of the axis angle
q1 = mat.axis_angle2quaternion([0.2,0.5,1], np.radians(45))
q2 = mat.axis_angle2quaternion([0.25,1,0.3], np.radians(30))
print('q1 =')
print(q1)
print('q2 =')
print(q2)
print('\n')


# 两个四元数乘积的模等于模的乘积
# ||q1*q2|| == ||q1||*||q2||
# The module of the product of two quaternions is equal to the product of modules
mul = math.quaternion_mul(q1,q2)
modulus = math.quaternion_modulus(mul)
print(modulus)

q1_m = math.quaternion_modulus(q1)
q2_m = math.quaternion_modulus(q2)
print(q1_m*q2_m)
print('\n')


# q*q_inv == q_inv*q == 1
q = q1
q_inv = math.quaternion_inverse(q)

q_q_inv = math.quaternion_mul(q,q_inv)
print(q_q_inv)
q_inv_q = math.quaternion_mul(q_inv,q)
print(q_inv_q)

