import numpy as np
import matplotlib.pyplot as plt


# 用于更好的显示矩阵
def printMatrix(matrix):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.figure.colorbar(im, ax=ax)
    plt.show()


# 矩阵转换，将上三角矩阵转换与对称矩阵互相转换
def transfer_matrix(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("矩阵必须为方阵")
    # 判断矩阵是对称矩阵还是上三角矩阵
    if np.allclose(matrix, np.triu(matrix)):
        # 如果是上三角矩阵，则转换为对称矩阵
        return matrix + matrix.T - np.diag(matrix.diagonal())
    else:
        # 如果是对称矩阵，则转换为上三角矩阵
        return np.triu(matrix)
