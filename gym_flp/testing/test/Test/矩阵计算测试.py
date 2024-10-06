# 矩阵计算测试

import numpy as np

np1 = np.array(
    [
        [0, 3, 2, 3],
        [0, 0, 4, 1],
        [0, 0, 0, 2],
        [0, 0, 0, 0],
    ]
)

np2 = np.array(
    [
        [0, 2, 1.5, 1.5],
        [2, 0, 1.5, 1.5],
        [1.5, 1.5, 0, 1],
        [1.5, 1.5, 1, 0],
    ]
)


np3 = np1 * np2
np4 = np.dot(np1, np2)
print(np3)
print(np.sum(np3))
print(np4)
print(np.sum(np4))
