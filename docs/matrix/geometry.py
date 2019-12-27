from pycontrol import matrix as mat
import numpy as np



# 沿Z轴旋转45度
# Rotate 45 degrees along the Z axis
rotate_matrix = mat.rotate_axis_angle([0,0,1], np.radians(45))
print('rotation matrix =')
print(rotate_matrix)
print('\n')

# 用旋转矩阵进行坐标变换
# Coordinate transformation with rotation matrix
v = np.array([1,0,0])
v_rotated = mat.rotate_R(rotate_matrix, v)
print('(1,0,0) after rotation =')
print(v_rotated)
print('\n')

# 将旋转矩阵转换为Z-Y-X欧拉角（逆解）
# Transform rotation matrix to z-y-x Euler angle (inverse solution)
euler_angles = mat.inverse_rotate_zyx(rotate_matrix)
print('yaw pitch roll =')
print(np.degrees(euler_angles))
print('\n')


# 由旋转矩阵和平移求变换矩阵
# Finding transformation matrix from rotation matrix and translation matrix
T = mat.Rt2T(rotate_matrix, [1,3,4])
print('transform matrix =')
print(T)

# 用变换矩阵进行坐标变换
# Coordinate transformation with transform matrix
v_transformed = mat.transform_homogeneous(T, v)
print('(1,0,0) after transform =')
print(v_transformed)
print('\n')


# 轴角转换为四元数
# Axis angle to quaternion
q = mat.axis_angle2quaternion([0,0,1], np.radians(45))
print('quaterniond from axis angle =')
print(q)
print('\n')

# 旋转矩阵转换为四元数
# rotation matrix to quaternion
q = mat.R2quaternion(rotate_matrix)
print('quaterniond from rotation matrix =')
print(q)
R = mat.quaternion2R(q)
print('rotation matrix from quaterniond =')
print(R)
print('\n')


# 用四元数进行坐标变换
# Coordinate transformation with quaternion
v_rotated = mat.rotate_quaternion(q, v)
print('(1,0,0) after quaternion rotation =')
print(v_rotated)



