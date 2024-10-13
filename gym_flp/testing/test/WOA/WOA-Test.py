# 鲸鱼优化算法解决x^2的路子i


# 定义目标函数：f(x) = x^2
import numpy as np


def objective_function(x):
    return x**2


# 鲸鱼优化算法（WOA）用于最小化 x^2
def woa_optimize(objective_function, lb, ub, dim, max_iter, n_whales):
    # 在边界内随机初始化鲸鱼位置
    whales = np.random.uniform(lb, ub, (n_whales, dim))
    # 计算所有鲸鱼的适应度（目标函数值）
    fitness = np.apply_along_axis(objective_function, 1, whales)

    # 记录到目前为止找到的最佳鲸鱼（领头鲸）
    leader_position = whales[np.argmin(fitness)]
    leader_score = min(fitness)

    # WOA算法主循环
    for t in range(max_iter):
        for i in range(n_whales):
            a = 2 - t * (2 / max_iter)  # 从2线性减少到0
            r = np.random.uniform(0, 1)  # 随机数 [0,1]
            A = 2 * a * r - a  # A向量
            C = 2 * np.random.uniform(0, 1)  # C向量

            # 探索阶段
            if np.random.uniform(0, 1) < 0.5:
                if abs(A) >= 1:
                    rand_whale = whales[
                        np.random.randint(0, n_whales)
                    ]  # 随机选择一只鲸鱼
                    D = abs(C * rand_whale - whales[i])  # 距离
                    whales[i] = rand_whale - A * D  # 更新位置
            # 开采阶段（收缩或螺旋）
            else:
                if abs(A) < 1:
                    D_leader = abs(C * leader_position - whales[i])  # 距离领头鲸的距离
                    whales[i] = leader_position - A * D_leader  # 更新位置
                else:
                    # 朝向领头鲸的螺旋更新
                    distance_to_leader = abs(leader_position - whales[i])
                    b = 1  # 定义螺旋形状的常数
                    l = np.random.uniform(-1, 1)  # 在[-1,1]中随机数
                    whales[i] = (
                        distance_to_leader * np.exp(b * l) * np.cos(2 * np.pi * l)
                        + leader_position
                    )

            # 边界检查
            whales[i] = np.clip(whales[i], lb, ub)

        # 重新计算所有鲸鱼的适应度
        fitness = np.apply_along_axis(objective_function, 1, whales)

        # 更新领头鲸
        min_fitness_idx = np.argmin(fitness)
        if fitness[min_fitness_idx] < leader_score:
            leader_score = fitness[min_fitness_idx]
            leader_position = whales[min_fitness_idx]

    return leader_position, leader_score


# 参数设置
lb = -10  # 下界
ub = 10  # 上界
dim = 1  # 问题维度（1D，对应x^2）
max_iter = 100  # 最大迭代次数
n_whales = 50  # 鲸鱼数量

# 运行鲸鱼优化算法
best_position, best_score = woa_optimize(
    objective_function, lb, ub, dim, max_iter, n_whales
)

print(f"最佳位置: {best_position}")
print(f"最佳适应度: {best_score}")
