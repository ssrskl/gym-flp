# 矩阵转移工具类，用于处理物流强度矩阵的问题

import os
import pickle
import numpy as np

# 取消长度限制
np.set_printoptions(threshold=np.inf)
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
print(__location__)
(
    problems,  # 问题模型
    FlowMatrices,  # 流量矩阵
    sizes,  # 尺寸数据
    LayoutWidths,  # 布局宽度 X W
    LayoutLengths,  # 布局长度 Y H
) = pickle.load(
    open(
        os.path.join(__location__, "maoyan_cont_instances.pkl"),
        "rb",
    )
)


def transfer_matrix(matrix: np.ndarray):
    """
    转置矩阵
    :param matrix: 矩阵
    :return: 转置后的矩阵
    """
    LowerTriangular = np.tril(matrix, -1).T
    resultMatrix = LowerTriangular + matrix
    resultMatrix = np.triu(resultMatrix)

    return resultMatrix


print("---------------------------------转换前---------------------------------")
print(FlowMatrices["SC35-maoyan"])
# 测试
print("---------------------------------转换后---------------------------------")

print(transfer_matrix(FlowMatrices["SC35-maoyan"]))
# 不换行打印
print("转换后的矩阵不换行打印:")
transfer_matrix = transfer_matrix(FlowMatrices["SC30"])
for row in transfer_matrix:
    print(' '.join(map(str, row)))



# 将转换后的矩阵保存到txt文件，中间用两个空格隔开

    
    


# # 创建一个5*5二维矩阵
# arrays = np.array(
#     [
#         [0, 0, 0, 1, 0],
#         [0, 0, 2, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 3, 6, 0, 0],
#     ]
# )
# # 得到下三角矩阵并转置
# triledArrays = np.tril(arrays, -1).T
# # 矩阵相加
# result = triledArrays + arrays
# # 对下三角矩阵清零
# result = np.triu(result)
# # 打印结果
# print(result)
