# 数据加载器
import os
import pickle

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
(
    problems,  # 问题模型
    FlowMatrices,  # 流量矩阵
    sizes,  # 尺寸数据
    LayoutWidths,  # 布局宽度
    LayoutLengths,  # 布局长度
) = pickle.load(
    open(
        os.path.join(__location__, "instances/continual", "cont_instances.pkl"),
        "rb",
    )
)
print(f"problems: {problems}")
print(f"FlowMatrices: {FlowMatrices}")
print(f"sizes: {sizes}")
print(f"LayoutWidths: {LayoutWidths}")
print(f"LayoutLengths: {LayoutLengths}")

# print(f"sizes: {sizes}")
