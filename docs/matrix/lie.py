from pycontrol import mat
import numpy as np



# 构造沿Z轴旋转90度的旋转矩阵
# Construction of rotation matrix rotating 90 degrees along Z axis
R = mat.rotate_axis_angle(np.array([0,0,1]), np.radians(90))
# 或者四元数
# Or quaternion
q = mat.R2quaternion(R)


# 从旋转矩阵构造李代数
# Constructing Lie algebra from rotation matrix
SO3 = mat.SO3_R(R)
print('SO3 =')
print(mat.SO3_matrix(SO3))

# 从四元数构造李代数
# Constructing Lie algebra from quaternion
SO3 = mat.SO3_q(q)
print('SO3 =')
print(mat.SO3_matrix(SO3))


# 使用对数映射获得李代数
# Using logarithmic mapping to obtain Lie algebra
so3 = mat.SO3_log(SO3)
print('so3 =')
print(so3)
print('\n')

# 向量到反对称矩阵
# Vector to antisymmetric matrix
so3_hat = mat.SO3_hat(so3)
print('so3 hat =')
print(so3_hat)
print('\n')

# 反对称矩阵到向量
# antisymmetric matrix to vector
so3_vee = mat.SO3_vee(so3_hat)
print('so3 hat vee =')
print(so3_vee)
print('\n')


# 增量扰动模型的更新
# Update of incremental disturbance model
update_so3 = np.array([1e-4,0,0])
so3_updated = mat.so3_exp_update(update_so3, R)
print("so3 updated =")
print(mat.SO3_matrix(so3_updated))
print('\n')
print('*'*20)


# 沿X轴平移1
# Translate 1 along X axis
t = np.array([1,0,0])


# 从R, t构造李代数
# Constructing Lie algebra from R, t
SE3 = mat.SE3_Rt(R, t)
print('SE3 =')
print(mat.SE3_matrix(SE3))

# 从q, t构造李代数
# Constructing Lie algebra from q, t
SE3 = mat.SE3_qt(q, t)
print('SE3 =')
print(mat.SE3_matrix(SE3))


# 旋转在前，平移在后
# Rotate in front, pan in back
se3 = mat.SE3_log(SE3)
print('se3 =')
print(se3)


# 向量到反对称矩阵
# Vector to antisymmetric matrix
se3_hat = mat.SE3_hat(se3)
print('se3 hat =')
print(se3_hat)
print('\n')


# 反对称矩阵到向量
# antisymmetric matrix to vector
se3_vee = mat.SE3_vee(se3_hat)
print('se3 hat vee =')
print(se3_vee)
print('\n')


# 增量扰动模型的更新
# Update of incremental disturbance model
update_se3 = np.array([1e-4,0,0, 0,0,0])
se3_updated = mat.se3_exp_update(update_se3, R, t)
print("se3 updated =")
print(mat.SE3_matrix(se3_updated))


