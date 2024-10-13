import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 创建逻辑图
fig, ax = plt.subplots(figsize=(10, 8))
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
# 画出步骤的矩形框
rects = [
    mpatches.Rectangle(
        (0.35, 0.85), 0.3, 0.1, edgecolor="black", facecolor="lightgray", lw=2
    ),
    mpatches.Rectangle(
        (0.35, 0.70), 0.3, 0.1, edgecolor="black", facecolor="lightgray", lw=2
    ),
    mpatches.Rectangle(
        (0.35, 0.55), 0.3, 0.1, edgecolor="black", facecolor="lightgray", lw=2
    ),
    mpatches.Rectangle(
        (0.35, 0.40), 0.3, 0.1, edgecolor="black", facecolor="lightgray", lw=2
    ),
    mpatches.Rectangle(
        (0.35, 0.25), 0.3, 0.1, edgecolor="black", facecolor="lightgray", lw=2
    ),
    mpatches.Rectangle(
        (0.35, 0.10), 0.3, 0.1, edgecolor="black", facecolor="lightgray", lw=2
    ),
]

for rect in rects:
    ax.add_patch(rect)

# 添加文本标签
steps = [
    "初始化鲸鱼位置",
    "计算适应度",
    "更新领头鲸",
    "迭代次数 < 最大迭代次数?",
    "更新鲸鱼位置",
    "输出最佳解",
]

for i, step in enumerate(steps):
    ax.text(
        0.5,
        0.9 - i * 0.15,
        step,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

# 添加箭头
arrows = [
    (0.5, 0.84, 0.5, 0.75),  # 从"初始化鲸鱼位置"到"计算适应度"
    (0.5, 0.69, 0.5, 0.60),  # 从"计算适应度"到"更新领头鲸"
    (0.5, 0.54, 0.5, 0.45),  # 从"更新领头鲸"到"迭代次数 < 最大迭代次数?"
    (0.5, 0.39, 0.5, 0.30),  # 从"迭代次数 < 最大迭代次数?"到"更新鲸鱼位置"
    (0.5, 0.24, 0.5, 0.15),  # 从"更新鲸鱼位置"到"输出最佳解"
]

for arrow in arrows:
    ax.annotate(
        "",
        xy=(arrow[2], arrow[3]),
        xytext=(arrow[0], arrow[1]),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )

# 添加"是"和"否"分支
ax.annotate(
    "是",
    xy=(0.65, 0.45),
    xytext=(0.65, 0.55),
    arrowprops=dict(facecolor="black", shrink=0.05),
)
ax.annotate(
    "否",
    xy=(0.35, 0.45),
    xytext=(0.35, 0.30),
    arrowprops=dict(facecolor="black", shrink=0.05),
)

# 画边界
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")  # 关闭坐标轴

# 显示逻辑图
plt.title("鲸鱼优化算法逻辑图", fontsize=16, fontweight="bold")
plt.show()
