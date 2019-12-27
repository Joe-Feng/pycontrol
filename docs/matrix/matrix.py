from pycontrol import matrix as mat
import numpy as np



# 随机数矩阵
# Random number matrix
matrix = np.random.uniform(0, 10, (3,3))
print('random matrix:')
print(matrix)
print('*'*20)

print('transpose: ', mat.transpose(matrix))       # 转置
print('sum: ', mat.sum(matrix))                   # 元素和
print('trace: ', mat.trace(matrix))               # 迹
print('inverse: ', mat.inverse(matrix))           # 逆
print('determinant: ', mat.determinant(matrix))   # 行列式
print('*'*20)


# 特征值和特征向量
print('eigen values and eigen vectors:')
print(mat.eigen(matrix)[0])
print(mat.eigen(matrix)[1])
print('*'*20)


# 求解方程 matrix_NN * X = b
# Solve equation matrix_NN * X = b
matrix_NN = np.random.uniform(0,10,(50,50))
v_Nd = np.random.uniform(0,10,(50,1))

solution = np.matmul(mat.inverse(matrix_NN), v_Nd)
print('solution:')
print(solution)
print(solution.shape)

