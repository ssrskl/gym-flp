import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号


# 定义设施的数据
labels = np.array(["A", "B", "C", "D", "E"])
positions = np.array(
    [
        [2, 3, 1, 2],  # (x, y, width, height)
        [5, 7, 1.5, 1],
        [8, 2, 2, 1],
        [4, 5, 1.2, 1.5],
        [6, 6, 1, 1.5],
    ]
)

# 创建图形和坐标轴
fig, ax = plt.subplots()

# 绘制长方形设施
for i, label in enumerate(labels):
    x, y, width, height = positions[i]
    # 创建长方形
    rect = Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        edgecolor="black",
        facecolor="blue",
        alpha=0.5,
    )
    ax.add_patch(rect)
    # 添加标签
    ax.text(x, y, label, fontsize=12, ha="center", va="center", color="white")

# 设置图形标题和标签
ax.set_title("设施布局图")
ax.set_xlabel("X 坐标")
ax.set_ylabel("Y 坐标")

# 设置坐标轴范围
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# 添加网格
plt.grid(True)
plt.gca().set_aspect("equal", adjustable="box")

# 显示图形
plt.show()
